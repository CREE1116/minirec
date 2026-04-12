import torch
import numpy as np
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class LAE(BaseModel):
    """
    LAE: Linear AutoEncoder (Wiener Filter form)
    Optimized with Hybrid CPU/GPU calculation.
    Formula: S = I - lambda * (X^T X + lambda * I)^-1
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.weight_matrix = None
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting LAE (lambda={self.reg_lambda}) on {self.device}...")
        
        # 1. Gram matrix G = X^T X (CPU Sparse -> Dense)
        X = get_train_matrix_scipy(data_loader)
        self.train_matrix_scipy = X

        print("  computing gram matrix (CPU)...")
        G_np = compute_gram_matrix(X, data_loader)
        
        # 2. Solve linear system on CPU to avoid VRAM issues
        print("  solving linear system (CPU)...")
        # A = G + lambda * I
        A_np = G_np.copy()
        A_np[np.diag_indices_from(A_np)] += self.reg_lambda
        
        # S = G @ (G + lambda*I)^-1 -> (G + lambda*I) S = G
        try:
            W_np = np.linalg.solve(A_np, G_np)
        except np.linalg.LinAlgError:
            print("[Warning] Singular matrix, using inverse with stronger regularization.")
            A_np[np.diag_indices_from(A_np)] += 1e-4
            W_np = np.linalg.solve(A_np, G_np)

        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        print("LAE fitting complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_matrix = self.train_matrix_scipy[users].toarray()
        input_tensor = torch.tensor(input_matrix, dtype=torch.float32, device=self.device)
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
