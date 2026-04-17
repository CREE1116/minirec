import torch
import numpy as np
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class DLAE(BaseModel):
    """
    DLAE: Denoising Linear AutoEncoder
    Optimized with NumPy inv and Strict float32.
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = np.float32(config['model'].get('reg_lambda', 100.0))
        self.dropout_p = np.float32(config['model'].get('dropout_p', 0.5))
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting DLAE (p={self.dropout_p}) on CPU using NumPy inv...")
        
        # 1. Load data
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr()

        print("  computing gram matrix (CPU float32)...")
        G_np = compute_gram_matrix(X_sp, data_loader)
        gc.collect()
        
        # 2. Add Dropout Penalty
        p = self.dropout_p
        dropout_penalty = ((p / (np.float32(1.0) - p)) * np.diag(G_np)).astype(np.float32)
        
        # 3. Solve via Inverse: W = inv(G + penalty + lambda*I) @ G
        print("  inverting matrix (NumPy float32)...")
        G_lhs = G_np.astype(np.float32)
        G_lhs[np.diag_indices_from(G_lhs)] += (dropout_penalty + self.reg_lambda)
        
        A_inv = np.linalg.inv(G_lhs).astype(np.float32)
        del G_lhs
        gc.collect()
        
        print("  finalizing weight matrix (matmul)...")
        # G_np is already the fresh original G from compute_gram_matrix
        W_np = (A_inv @ G_np).astype(np.float32)
        
        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        del A_inv, W_np, G_np, dropout_penalty

        gc.collect()
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        print("DLAE fitting complete.")

    def forward(self, user_indices):
        """Memory-efficient hybrid inference"""
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
