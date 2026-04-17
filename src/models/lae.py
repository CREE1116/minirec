import torch
import numpy as np
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class LAE(BaseModel):
    """
    LAE: Linear AutoEncoder (Wiener Filter form)
    Formula: S = G @ (G + lambda * I)^-1
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting LAE (lambda={self.reg_lambda}) on CPU/GPU Hybrid...")
        
        # 1. Load data and store for hybrid inference
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr()

        print("  computing gram matrix (CPU)...")
        G_np = compute_gram_matrix(X_sp, data_loader)
        
        # 2. Solve linear system
        print("  solving linear system (CPU)...")
        A_np = G_np.copy()
        A_np[np.diag_indices_from(A_np)] += self.reg_lambda
        
        try:
            W_np = np.linalg.solve(A_np, G_np)
        except np.linalg.LinAlgError:
            print("[Warning] Singular matrix, using stronger regularization.")
            A_np[np.diag_indices_from(A_np)] += 1e-4
            W_np = np.linalg.solve(A_np, G_np)

        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        print("LAE fitting complete.")

    def forward(self, user_indices):
        """Memory-efficient hybrid inference"""
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
