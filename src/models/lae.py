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
        self.reg_lambda = np.float32(config['model'].get('reg_lambda', 500.0))
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting LAE (lambda={self.reg_lambda}) on CPU (Minimal Copy)...")
        
        # 1. Load data
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr()

        print("  computing gram matrix (CPU float32)...")
        G_np = compute_gram_matrix(X_sp, data_loader)
        gc.collect()
        
        # 2. Solve via Inverse: W = inv(G + lambda*I) @ G
        print("  inverting matrix (NumPy)...")
        A_np = G_np.copy() # Need to keep G_np for matmul later
        A_np[np.diag_indices_from(A_np)] += self.reg_lambda
        
        A_inv = np.linalg.inv(A_np)
        del A_np
        gc.collect()
        
        # W = A_inv @ G
        print("  finalizing weight matrix (matmul)...")
        W_np = (A_inv @ G_np)
        
        # 3. Transfer to GPU and clean up
        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        del A_inv, W_np, G_np
        gc.collect()

        if 'cuda' in str(self.device): torch.cuda.empty_cache()
        print("LAE fitting complete.")

    def forward(self, user_indices):
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
