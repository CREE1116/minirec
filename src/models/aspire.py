import torch
import numpy as np
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class Aspire(BaseModel):
    """
    ASPIRE: Spectral Purification via Signal-to-Noise Ratio (d^2/S)
    Optimized with Hybrid CPU/GPU calculation.
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

        # ── 1. Gram matrix G = X^T X (CPU) ──────────────────────────────────
        print("  computing gram matrix (CPU)...")
        G_np = compute_gram_matrix(X, data_loader)
        
        # ── 2. Perform SNR and Scaling on CPU ──────────────────────────
        d = G_np.diagonal()       # n_i
        S = G_np.sum(axis=1)      # S_i

        log_lambda = 2.0 * np.log(d + self.eps) - np.log(S + self.eps)
        log_lambda_centered = log_lambda - log_lambda.mean()
        reliability = np.exp(log_lambda_centered)

        scale_factor = np.power(reliability + self.eps, -self.alpha / 2.0)
        G_tilde = G_np * scale_factor[:, np.newaxis] * scale_factor[np.newaxis, :]

        # ── 3. Ridge Regression (CPU Inversion) ─────────────────────────────
        print("  inverting matrix (CPU)...")
        A_np = G_tilde.copy()
        A_np[np.diag_indices_from(A_np)] += self.reg_lambda
        
        P_np = np.linalg.inv(A_np)
        
        # Final weights: B_{ij} = -P_{ij} / P_{jj} (i != j), B_{ii} = 0
        diag_P = np.diag(P_np)
        B_np = P_np / (-diag_P[:, np.newaxis] + self.eps)
        np.fill_diagonal(B_np, 0)
        
        self.weight_matrix = torch.tensor(B_np, dtype=torch.float32, device=self.device)
        print("Aspire fitting complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_matrix = self.train_matrix_scipy[users].toarray()
        input_tensor = torch.tensor(input_matrix, dtype=torch.float32, device=self.device)
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
