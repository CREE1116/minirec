import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy

class CausalAspire(BaseModel):
    """
    Causal ASPIRE (ASPIRE-Core)
    - Stage 1: User-side Fractional IPW (beta) -> Variance Stabilization
    - Stage 2: Item-side Geometric Ensemble Normalization (alpha) -> Causal Ensemble
    - Stage 3: Ridge Regression via Solve -> Faster and more stable
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 10.0) 
        self.alpha      = config['model'].get('alpha', 0.5)
        self.beta       = config['model'].get('beta', 0.5)
        self.eps        = 1e-12
        self.weight_matrix = None
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting Causal ASPIRE (lambda={self.reg_lambda}, alpha={self.alpha}, beta={self.beta}) on {self.device}...")

        # 1. Load data
        self.train_matrix_scipy = get_train_matrix_scipy(data_loader)
        X_sp = self.train_matrix_scipy

        n_u = np.asarray(X_sp.sum(axis=1)).ravel()  # (U,)
        K   = X_sp.shape[1]

        # ── Stage 1: User-side Fractional IPW (CPU Sparse) ───────────────────────────
        print("  stage 1: computing user-side fractional IPW gram matrix...")
        user_weights = np.power(n_u + self.eps, -self.beta)
        D_U_inv = sp.diags(user_weights)

        X_weighted = D_U_inv @ X_sp                 
        G_U = (X_sp.T @ X_weighted).toarray()       

        # ── Stage 2: Item-side Geometric Ensemble Normalization ──────────────────
        print("  stage 2: applying item-side geometric ensemble normalization...")
        A_i = G_U.diagonal().copy()
        scale = np.power(A_i + self.eps, -self.alpha / 2.0)
        G_tilde = G_U * scale[:, None] * scale[None, :]

        # ── Stage 3: Ridge Regression via Solve (GPU) ────────────────
        print(f"  stage 3: solving ridge system on {self.device}...")
        G_torch = torch.tensor(G_tilde, dtype=torch.float32, device=self.device)
        A_mat = G_torch + self.reg_lambda * torch.eye(K, device=self.device)

        try:
            # W = (G_tilde + lambda*I)^{-1} @ G_tilde
            W = torch.linalg.solve(A_mat, G_torch)
        except (torch._C._LinAlgError, RuntimeError):
            print("[Warning] Singular matrix, applying stronger regularization.")
            A_mat.diagonal().add_(self.reg_lambda * 10 + 1e-4)
            W = torch.linalg.solve(A_mat, G_torch)

        # Post-masking (Prevent self-recommendation)
        # W.diagonal().zero_()

        self.weight_matrix = W
        print("Causal ASPIRE fitting complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_tensor = torch.tensor(self.train_matrix_scipy[users].toarray(), dtype=torch.float32, device=self.device)
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None