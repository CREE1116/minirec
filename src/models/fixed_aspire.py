import torch
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class FixedAspire(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = np.float32(config['model'].get('reg_lambda', 10.0))
        self.alpha      = np.float32(config['model'].get('alpha', 1.0))
        self.eps        = np.float32(1e-12)
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting ASPIRE (reg_lambda={self.reg_lambda}, alpha={self.alpha}) on CPU...")

        X_sp = get_train_matrix_scipy(data_loader) # (U, I) sparse
        self.train_matrix_cpu = X_sp.tocsr() # Store for hybrid inference

        # ── Step 1: Pre-calculate Weights ──
        n_u = np.asarray(X_sp.sum(axis=1)).ravel().astype(np.float32)
        user_weights = (np.float32(1.0) / (np.power(n_u, self.alpha) + self.eps)).astype(np.float32)

        n_i = np.asarray(X_sp.sum(axis=0)).ravel().astype(np.float32)
        item_weights = np.power(n_i + self.eps, -self.alpha/np.float32(2.0)).astype(np.float32)

        # ── Step 2: Optimized Block-wise Gram Matrix ──
        print("  Constructing G_tilde (CPU Block-wise Dense)...")
        # This utility avoids the 83GB sparse-sparse peak
        G_tilde_np = compute_gram_matrix(X_sp, data_loader, 
                                         weights=user_weights, 
                                         item_weights=item_weights)
        
        del n_u, user_weights, n_i, item_weights
        gc.collect()

        # ── Step 3: EASE solver (CPU NumPy In-place) ──
        print("  Solving EASE closed-form (CPU In-place float32)...")
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
        print("ASPIRE fitting complete.")

    def forward(self, user_indices):
        return self._get_batch_ratings(user_indices, self.weight_matrix)
