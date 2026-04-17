import torch
import numpy as np
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class EASE(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting EASE (lambda={self.reg_lambda}) on CPU...")
        
        # Use shared utility for efficient sparse matrix loading
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr() # Store on CPU for hybrid inference

        print("  computing gram matrix (CPU)...")
        G_np = compute_gram_matrix(X_sp, data_loader)
        
        print("  inverting matrix (CPU)...")
        G_np[np.diag_indices_from(G_np)] += self.reg_lambda
        
        P_np = np.linalg.inv(G_np)
        
        # Final weights: B_{ij} = -P_{ij} / P_{jj} (i != j), B_{ii} = 0
        diag_P = np.diag(P_np)
        B_np = -P_np / (diag_P + 1e-12)
        np.fill_diagonal(B_np, 0)
        
        self.weight_matrix = torch.tensor(B_np, dtype=torch.float32, device=self.device)
        print("EASE fitting complete.")

    def forward(self, user_indices):
        """Memory-efficient hybrid inference"""
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
