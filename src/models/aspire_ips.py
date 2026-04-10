import torch
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix
import numpy as np

class AspireIPS(BaseModel):
    """
    ASPIRE-IPS: Standard Ridge Regression with IPS-based Scaling
    - n_u: 유저별 활동량 (X의 행합)
    - A_i: 아이템별 IPS 보정 인기도 (X.multiply(1/n_u).sum(axis=0))
    - G_tilde = diag(A)^{-alpha/2} * G * diag(A)^{-alpha/2}
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.alpha      = config['model'].get('alpha', 0.5) 
        self.eps        = 1e-12
        self.weight_matrix = None
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting AspireIPS (lambda={self.reg_lambda}, alpha={self.alpha}) on {self.device}...")

        X = get_train_matrix_scipy(data_loader)
        self.train_matrix_scipy = X

        # ── 1. Gram matrix G = X^T X (CPU) ──────────────────────────────────
        print("  computing gram matrix (CPU)...")
        G_np = compute_gram_matrix(X, data_loader)

        # ── 2. Calculate IPS-based Item Popularity ───────────────────────────
        # n_u: 유저별 활동량 (X의 행합)
        n_u = np.array(X.sum(axis=1)).flatten()

        # A_i: 아이템별 IPS 보정 인기도
        inv_nu = 1.0 / (n_u + self.eps)
        X_weighted = X.multiply(inv_nu[:, None])
        A = np.array(X_weighted.sum(axis=0)).flatten()

        # ── 3. G_tilde = diag(A)^{-α/2} * G * diag(A)^{-α/2} ─────────────────
        scale_factor = np.power(A + self.eps, -self.alpha / 2.0)
        G_tilde = G_np * scale_factor[:, None] * scale_factor[None, :]

        # ── 4. Standard Ridge Regression (using solve) ────────────────────────
        print(f"  solving ridge solution on {self.device}...")
        
        G_tilde_t = torch.tensor(G_tilde, dtype=torch.float32, device=self.device)
        A_t = G_tilde_t.clone()
        A_t.diagonal().add_(self.reg_lambda)

        # W = (G_tilde + lambda*I)^{-1} @ G_tilde
        self.weight_matrix = torch.linalg.solve(A_t, G_tilde_t)

        print("AspireIPS fitting complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_matrix = self.train_matrix_scipy[users].toarray()
        input_tensor = torch.tensor(input_matrix, dtype=torch.float32, device=self.device)

        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
