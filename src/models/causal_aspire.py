import torch
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class CausalAspire(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = np.float32(config['model'].get('reg_lambda', 10.0))
        self.alpha      = np.float32(config['model'].get('alpha', 1.0))
        self.eps        = np.float32(1e-12)
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting Causal ASPIRE (reg_lambda={self.reg_lambda}, alpha={self.alpha}) on CPU...")

        # 1. Load data
        X_sp = get_train_matrix_scipy(data_loader) # (U, I) sparse
        self.train_matrix_cpu = X_sp.tocsr() # Hybrid inference

        # ── Step 1: User-side Propensity (q_u) ──
        n_u = np.asarray(X_sp.sum(axis=1)).ravel().astype(np.float32)
        user_weights = (np.float32(1.0) / (np.power(n_u, self.alpha) + self.eps)).astype(np.float32)
        D_U_inv = sp.diags(user_weights, dtype=np.float32)

        # ── Step 2: Item-side Bias (b_i) ──
        print("  Computing item-side standardized bias (float32)...")
        item_bias = np.asarray(X_sp.T.dot(user_weights)).ravel().astype(np.float32)

        # ── Step 3: Gram Matrix Standardization ──
        print("  Standardizing gram matrix (CPU Sparse)...")
        item_weights = (np.float32(1.0) / np.sqrt(item_bias + self.eps)).astype(np.float32)
        D_I_inv_half = sp.diags(item_weights, dtype=np.float32)

        # G_U = X.T @ D_U^{-1} @ X
        X_scaled = D_U_inv @ X_sp
        G_U_sp = X_sp.T @ X_scaled # (I, I) sparse
        
        # G_tilde = D_I^{-0.5} @ G_U @ D_I^{-0.5}
        G_tilde_sp = D_I_inv_half @ G_U_sp @ D_I_inv_half
        G_tilde_np = G_tilde_sp.toarray().astype(np.float32)
        
        del D_U_inv, X_scaled, G_U_sp, G_tilde_sp, item_weights, user_weights, item_bias, n_u
        gc.collect()

        # ── Step 4: Strict EASE Solution (CPU NumPy float32) ──
        print("  Solving strict EASE closed-form (CPU In-place float32)...")
        G_tilde_np[np.diag_indices_from(G_tilde_np)] += self.reg_lambda
        
        P_np = la.inv(G_tilde_np, overwrite_a=True).astype(np.float32)
        del G_tilde_np
        gc.collect()

        P_diag = np.diag(P_np).astype(np.float32)
        W_np = (-P_np / (P_diag[np.newaxis, :] + self.eps)).astype(np.float32)
        np.fill_diagonal(W_np, 0)

        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        del P_np, W_np
        
        gc.collect()
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        print("Causal ASPIRE fitting complete.")

    def forward(self, user_indices):
        return self._get_batch_ratings(user_indices, self.weight_matrix)
