import torch
import numpy as np
import gc
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
        print(f"Fitting LAE (lambda={self.reg_lambda}) on CPU...")
        
        # 1. Load data onto CPU
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr()

        print("  computing gram matrix (CPU)...")
        G_np = compute_gram_matrix(X_sp, data_loader)
        
        # 2. Solve linear system on CPU
        print("  solving linear system (CPU)...")
        # G_np is already a fresh copy from compute_gram_matrix
        A_np = G_np.astype(np.float32)
        A_np[np.diag_indices_from(A_np)] += self.reg_lambda
        
        # G_np를 보존해야 하므로, compute_gram_matrix에서 하나 더 가져옴 (Sparse 캐시 활용)
        G_orig = compute_gram_matrix(X_sp, data_loader).astype(np.float32) 

        try:
            W_np = np.linalg.solve(A_np, G_orig).astype(np.float32)
        except np.linalg.LinAlgError:
            print("[Warning] Singular matrix, using stronger regularization.")
            A_np[np.diag_indices_from(A_np)] += np.float32(1e-4)
            W_np = np.linalg.solve(A_np, G_orig).astype(np.float32)
        
        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        del A_np, G_orig, W_np, G_np

        gc.collect()
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        print("LAE fitting complete.")

    def forward(self, user_indices):
        """Memory-efficient hybrid inference"""
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
