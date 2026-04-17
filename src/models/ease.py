import torch
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class EASE(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = np.float32(config['model'].get('reg_lambda', 500.0))
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting EASE (lambda={self.reg_lambda}) on CPU...")
        
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr()

        print("  computing gram matrix (CPU float32)...")
        G_np = compute_gram_matrix(X_sp, data_loader).astype(np.float32)
        
        print("  inverting matrix (CPU In-place NumPy float32)...")
        G_np[np.diag_indices_from(G_np)] += self.reg_lambda
        
        # scipy.linalg.inv with overwrite_a=True saves 1 full matrix copy (8.3GB)
        P_np = la.inv(G_np, overwrite_a=True).astype(np.float32)
        del G_np
        gc.collect()
        
        diag_P = np.diag(P_np).astype(np.float32)
        B_np = (-P_np / (diag_P + np.float32(1e-12))).astype(np.float32)
        np.fill_diagonal(B_np, 0)
        
        self.weight_matrix = torch.tensor(B_np, dtype=torch.float32, device=self.device)
        del P_np, B_np
        
        gc.collect()
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        print("EASE fitting complete.")

    def forward(self, user_indices):
        """Memory-efficient hybrid inference"""
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
