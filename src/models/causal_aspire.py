import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy

class CausalAspire(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 10.0) 
        self.alpha      = config['model'].get('alpha', 1.0) # User-side scaling exponent
        self.eps        = 1e-12
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting Causal ASPIRE (reg_lambda={self.reg_lambda}, alpha={self.alpha})...")

        # 1. Load data
        X_sp = get_train_matrix_scipy(data_loader) # (U, I) sparse

        # ── Step 1: User-side Propensity (q_u) ───────────────────────────
        n_u = np.asarray(X_sp.sum(axis=1)).ravel()
        # q_u = n_u^alpha
        user_weights = 1.0 / (np.power(n_u, self.alpha) + self.eps)
        D_U_inv = sp.diags(user_weights)

        # ── Step 2: Item-side Bias (b_i) ─────────────────────────────────
        print("  Computing item-side standardized bias...")
        # b_i = sum_u (X_ui^2 / q_u)
        item_bias = np.asarray(X_sp.T.dot(user_weights)).ravel()

        # ── Step 3: Gram Matrix Standardization ──────────────────────────
        print("  Standardizing gram matrix...")
        item_weights = 1.0 / np.sqrt(item_bias + self.eps)
        D_I_inv_half = sp.diags(item_weights)

        # G_U = X.T @ D_U^{-1} @ X
        X_scaled = D_U_inv @ X_sp
        G_U_sp = X_sp.T @ X_scaled # (I, I) sparse

        # G_tilde = D_I^{-0.5} @ G_U @ D_I^{-0.5}
        G_tilde_sp = D_I_inv_half @ G_U_sp @ D_I_inv_half
        G_tilde = G_tilde_sp.toarray()

        # ── Step 4: Strict EASE Solution (CPU) ──────────────────────────
        print("  Solving strict EASE closed-form...")
        G_tilde[np.diag_indices_from(G_tilde)] += self.reg_lambda
        P_np = np.linalg.inv(G_tilde)

        P_diag = np.diag(P_np)
        W_np = -P_np / (P_diag[np.newaxis, :] + self.eps)
        np.fill_diagonal(W_np, 0)

        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        self.train_matrix_gpu = self.get_train_matrix(data_loader)
        print("Causal ASPIRE fitting complete.")

    def forward(self, user_indices):
        input_tensor = torch.index_select(self.train_matrix_gpu, 0, user_indices).to_dense()
        return input_tensor @ self.weight_matrix