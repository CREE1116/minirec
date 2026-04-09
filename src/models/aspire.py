import torch
import numpy as nn
import numpy as np
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class Aspire(BaseModel):
    """
    ASPIRE: Spectral Purification via Signal-to-Noise Ratio (d^2/S)
    optimized with Scipy Gram matrix calculation.
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.alpha      = config['model'].get('alpha', 1.0)
        self.eps        = 1e-12
        self.weight_matrix = None
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting Aspire (lambda={self.reg_lambda}, alpha={self.alpha}) on {self.device}...")
        
        X = get_train_matrix_scipy(data_loader)
        self.train_matrix_scipy = X

        # ── 1. Gram matrix G = X^T X ──────────────────────────────────────
        print("  computing gram matrix...")
        G = compute_gram_matrix(X)
        d = G.diagonal()       # n_i: 아이템 자기 에너지
        S = G.sum(axis=1)      # S_i: 그램 행합

        # ── 2. Log-space robust SNR ────────────────────────────────────────
        log_lambda = 2.0 * np.log(d + self.eps) - np.log(S + self.eps)
        log_lambda_centered = log_lambda - log_lambda.mean()
        reliability = np.exp(log_lambda_centered)

        # ── 3. G_tilde = Λ^{-α/2} G Λ^{-α/2} ────────────────────────────
        scale_factor = np.power(reliability + self.eps, -self.alpha / 2.0)
        G_tilde = G * scale_factor[:, None] * scale_factor[None, :]

        # ── 4. Standard Ridge Regression (EASE style) ──────────────
        print("  solving/inverting matrix...")
        A = G_tilde.copy()
        A[np.diag_indices(self.n_items)] += self.reg_lambda
        
        # P = (G_tilde + λI)^{-1}
        P = np.linalg.inv(A)
        B = P / (-np.diag(P))
        np.fill_diagonal(B, 0)
        
        self.weight_matrix = torch.tensor(B, dtype=torch.float32, device=self.device)
        print("Aspire fitting complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_matrix = self.train_matrix_scipy[users].toarray()
        input_tensor = torch.tensor(input_matrix, dtype=torch.float32, device=self.device)
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
