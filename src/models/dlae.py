import torch
import numpy as np
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class DLAE(BaseModel):
    """
    DLAE: Denoising Linear AutoEncoder
    Optimized with Hybrid CPU/GPU inference.
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.dropout_p = config['model'].get('dropout_p', 0.5)
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting DLAE (p={self.dropout_p}, lambda={self.reg_lambda}) on CPU/GPU Hybrid...")
        
        # 1. Load data onto CPU
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr()

        print("  computing gram matrix (CPU)...")
        G_np = compute_gram_matrix(X_sp, data_loader)
        
        # 2. Add Dropout Penalty: diag += (p/(1-p)) * G_jj
        p = min(self.dropout_p, 0.99)
        dropout_penalty = (p / (1.0 - p)) * np.diag(G_np)
        
        G_lhs = G_np.copy()
        G_lhs[np.diag_indices_from(G_lhs)] += (dropout_penalty + self.reg_lambda)
        
        print("  solving linear system (CPU)...")
        W_np = np.linalg.solve(G_lhs, G_np)
        
        # 3. Store weight matrix on GPU
        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        print("DLAE fitting complete.")

    def forward(self, user_indices):
        """Memory-efficient hybrid inference"""
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
