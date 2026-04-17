import torch
import numpy as np
import scipy.sparse as sp
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy

class FixedAspire(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 10.0)
        self.alpha      = config['model'].get('alpha', 1.0) # User-side scaling exponent
        self.eps = 1e-12
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting ASPIRE (reg_lambda={self.reg_lambda}, alpha={self.alpha}) on {self.device}...")

        X_sp = get_train_matrix_scipy(data_loader) # (U, I) sparse
        self.train_matrix_cpu = X_sp.tocsr() # Store for hybrid inference

        # ── Step 1 & 2: Scaling vectors ─────
        n_u = np.asarray(X_sp.sum(axis=1)).ravel()
        user_weights = np.power(n_u + self.eps, -self.alpha)
        D_U_inv = sp.diags(user_weights)

        n_i = np.asarray(X_sp.sum(axis=0)).ravel()
        item_weights = np.power(n_i + self.eps, -self.alpha/2.0)
        D_I_inv_half = sp.diags(item_weights)

        # ── Step 3: Normalized Gram Matrix ─────────────────────────────
        print("  Constructing G_tilde (CPU Sparse)...")
        X_scaled = D_U_inv @ X_sp
        G_raw_sp = X_sp.T @ X_scaled # (I, I) sparse

        G_tilde_sp = D_I_inv_half @ G_raw_sp @ D_I_inv_half
        G_tilde_np = G_tilde_sp.toarray().astype(np.float32)
        
        del D_U_inv, X_scaled, G_raw_sp, G_tilde_sp
        gc.collect()

        # ── Step 4: EASE solver ────────────────────────────────────────
        if 'cuda' in str(self.device) and G_tilde_np.shape[0] < 20000:
            print("  Solving EASE closed-form (GPU)...")
            G_torch = torch.from_numpy(G_tilde_np).to(self.device)
            del G_tilde_np
            gc.collect()

            G_torch.diagonal().add_(self.reg_lambda)
            P_torch = torch.linalg.inv(G_torch)
            del G_torch
            
            P_diag = torch.diagonal(P_torch)
            self.weight_matrix = -P_torch / (P_diag.unsqueeze(0) + self.eps)
            self.weight_matrix.diagonal().zero_()
            del P_torch
        else:
            print("  Solving EASE closed-form (CPU)...")
            G_tilde_np[np.diag_indices_from(G_tilde_np)] += self.reg_lambda
            P_np = np.linalg.inv(G_tilde_np)
            del G_tilde_np
            gc.collect()

            P_diag = np.diag(P_np)
            W_np = -P_np / (P_diag[np.newaxis, :] + self.eps)
            np.fill_diagonal(W_np, 0)

            self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
            del P_np, W_np

        gc.collect()
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        print("ASPIRE fitting complete.")

    def forward(self, user_indices):
        return self._get_batch_ratings(user_indices, self.weight_matrix)