import torch
import numpy as np
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class RLAE(BaseModel):
    """
    Relaxed Linear AutoEncoder (RLAE)
    Optimized with NumPy inv and Strict float32.
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = np.float32(config['model'].get('reg_lambda', 500.0))
        self.b = np.float32(config['model'].get('b', 0.0))
        self.eps = np.float32(1e-12)
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting RLAE (lambda={self.reg_lambda}, b={self.b}) on CPU using NumPy inv...")
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr()
        G_np = compute_gram_matrix(X_sp, data_loader).astype(np.float32)
        
        # G_np is already a fresh copy
        A = G_np
        A[np.diag_indices_from(A)] += self.reg_lambda
        
        print(f"  inverting matrix (NumPy float32)...")
        P = np.linalg.inv(A).astype(np.float32)
        del A
        gc.collect()
        
        diag_P = np.diag(P).astype(np.float32)
        penalty = np.maximum(self.reg_lambda, (np.float32(1.0) - self.b) / (diag_P + self.eps)).astype(np.float32)
        
        W = (-P * penalty[np.newaxis, :]).astype(np.float32)
        W[np.diag_indices_from(W)] += np.float32(1.0)
        
        self.weight_matrix = torch.tensor(W, dtype=torch.float32, device=self.device)
        del P, W
        
        gc.collect()
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        print("RLAE fitting complete.")

    def forward(self, user_indices):
        """Memory-efficient hybrid inference"""
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None

class RDLAE(BaseModel):
    """
    Relaxed Denoising Linear AutoEncoder (RDLAE)
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = np.float32(config['model'].get('reg_lambda', 500.0))
        self.dropout_p = np.float32(config['model'].get('dropout_p', 0.5))
        self.b = np.float32(config['model'].get('b', 0.0))
        self.eps = np.float32(1e-12)
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting RDLAE (lambda={self.reg_lambda}) on CPU using NumPy inv...")
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr()
        G_np = compute_gram_matrix(X_sp, data_loader).astype(np.float32)
        
        p = self.dropout_p
        dropout_penalty_np = ((p / (np.float32(1.0) - p)) * np.diag(G_np)).astype(np.float32)
        lambda_diag_np = (dropout_penalty_np + self.reg_lambda).astype(np.float32)
        
        # G_np is already a fresh copy
        A = G_np
        A[np.diag_indices_from(A)] += lambda_diag_np
        
        print(f"  inverting matrix (NumPy float32)...")
        P = np.linalg.inv(A).astype(np.float32)
        del A
        gc.collect()
        
        diag_P = np.diag(P).astype(np.float32)
        total_penalty = np.maximum(lambda_diag_np, (np.float32(1.0) - self.b) / (diag_P + self.eps)).astype(np.float32)
        
        W = (-P * total_penalty[np.newaxis, :]).astype(np.float32)
        W[np.diag_indices_from(W)] += np.float32(1.0)
        
        self.weight_matrix = torch.tensor(W, dtype=torch.float32, device=self.device)
        del P, W
            
        gc.collect()
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        print("RDLAE fitting complete.")
