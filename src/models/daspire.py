import torch
import numpy as np
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class DAspire(BaseModel):
    """
    DAspire: Spectral Purification via SNR with DLAE-style regularization.
    Optimized with Hybrid CPU/GPU calculation.
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.alpha      = config['model'].get('alpha', 1.0)
        self.dropout_p  = config['model'].get('dropout_p', 0.5)
        self.eps        = 1e-12
        self.weight_matrix = None
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting DAspire (lambda={self.reg_lambda}, alpha={self.alpha}, p={self.dropout_p}) on {self.device}...")

        X = get_train_matrix_scipy(data_loader)
        self.train_matrix_scipy = X

        # ── 1. Gram matrix G = X^T X (CPU) ──────────────────────────────────
        print("  computing gram matrix (CPU)...")
        G_np = compute_gram_matrix(X, data_loader)

        # ── 2. Move to GPU for fast SNR and Scaling ──────────────────────────
        G = torch.tensor(G_np, dtype=torch.float32, device=self.device)
        d = G.diagonal()
        S = G.sum(axis=1)

        log_lambda = 2.0 * torch.log(d + self.eps) - torch.log(S + self.eps)
        log_lambda_centered = log_lambda - log_lambda.mean()
        reliability = torch.exp(log_lambda_centered)

        scale_factor = torch.pow(reliability + self.eps, -self.alpha / 2.0)
        G_tilde = G * scale_factor.view(-1, 1) * scale_factor.view(1, -1)

        # ── 3. DLAE-style Diagonal Scaling ──────────────────────────────
        p = min(self.dropout_p, 0.99)
        w = (p / (1.0 - p)) * d

        G_lhs = G_tilde.clone()
        G_lhs.diagonal().add_(w + self.reg_lambda)

        print(f"  solving linear system on {self.device}...")
        self.weight_matrix = torch.linalg.solve(G_lhs, G_tilde)
        print("DAspire fitting complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_matrix = self.train_matrix_scipy[users].toarray()
        input_tensor = torch.tensor(input_matrix, dtype=torch.float32, device=self.device)
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None

