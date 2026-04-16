import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy

class FixedAspire(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 10.0)
        self.alpha      = config['model'].get('alpha', 1.0) # User-side scaling exponent (not used in FixedAspire but kept for consistency)
        self.eps = 1e-12
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting ASPIRE (reg_lambda={self.reg_lambda})...")

        X_sp = get_train_matrix_scipy(data_loader) # (U, I) sparse

        # ── Step 1 & 2: Scaling vectors (Original ASPIRE exponents) ─────
        n_u = np.asarray(X_sp.sum(axis=1)).ravel()
        user_weights = np.power(n_u + self.eps, -self.alpha)  # Original: -0.5
        D_U_inv = sp.diags(user_weights)

        n_i = np.asarray(X_sp.sum(axis=0)).ravel()
        item_weights = np.power(n_i + self.eps, -self.alpha/2.0) # Original: -0.25
        D_I_inv_half = sp.diags(item_weights)

        # ── Step 3: Normalized Gram Matrix ─────────────────────────────
        print("  Constructing G_tilde...")
        X_scaled = D_U_inv @ X_sp
        G_raw_sp = X_sp.T @ X_scaled # (I, I) sparse

        G_tilde_sp = D_I_inv_half @ G_raw_sp @ D_I_inv_half
        G_tilde = G_tilde_sp.toarray()

        # ── Step 4: EASE solver ────────────────────────────────────────
        print("  Solving EASE closed-form...")
        G_tilde[np.diag_indices_from(G_tilde)] += self.reg_lambda
        P_np = np.linalg.inv(G_tilde)

        P_diag = np.diag(P_np)
        W_np = -P_np / (P_diag[np.newaxis, :] + self.eps)
        np.fill_diagonal(W_np, 0)

        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        self.train_matrix_gpu = self.get_train_matrix(data_loader)
        print("ASPIRE fitting complete.")

    def forward(self, user_indices):
        input_tensor = torch.index_select(self.train_matrix_gpu, 0, user_indices).to_dense()
        return input_tensor @ self.weight_matrix