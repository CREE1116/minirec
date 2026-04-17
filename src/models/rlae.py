import torch
import numpy as np
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class RLAE(BaseModel):
    """
    Relaxed Linear AutoEncoder (RLAE)
    - Optimization: min ||X - XB||^2 + lambda ||B||^2  s.t.  diag(B) <= b
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.b = config['model'].get('b', 0.0)
        self.eps = 1e-12
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting RLAE (lambda={self.reg_lambda}, b={self.b}) on {self.device}...")
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr()
        G_np = compute_gram_matrix(X_sp, data_loader, device=self.device)
        
        if 'cuda' in str(self.device) and G_np.shape[0] < 20000:
            print("  inverting matrix (GPU)...")
            G_torch = torch.from_numpy(G_np).to(self.device)
            del G_np
            gc.collect()

            A = G_torch.clone()
            A.diagonal().add_(self.reg_lambda)
            P = torch.linalg.inv(A)
            del A
            
            diag_P = torch.diagonal(P)
            penalty = torch.maximum(torch.tensor(self.reg_lambda, device=self.device), (1.0 - self.b) / (diag_P + self.eps))
            
            self.weight_matrix = - (P * penalty.unsqueeze(0))
            self.weight_matrix.diagonal().add_(1.0)
            del P
        else:
            print("  inverting matrix (CPU)...")
            # G_np is already a fresh copy
            A = G_np
            A[np.diag_indices_from(A)] += self.reg_lambda
            
            P = np.linalg.inv(A)
            del A
            
            diag_P = np.diag(P)
            penalty = np.maximum(self.reg_lambda, (1.0 - self.b) / (diag_P + self.eps))
            
            W = - (P * penalty[np.newaxis, :])
            W[np.diag_indices_from(W)] += 1.0
            
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
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.dropout_p = config['model'].get('dropout_p', 0.5)
        self.b = config['model'].get('b', 0.0)
        self.eps = 1e-12
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting RDLAE (lambda={self.reg_lambda}, p={self.dropout_p}, b={self.b}) on {self.device}...")
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr()
        G_np = compute_gram_matrix(X_sp, data_loader, device=self.device)
        
        p = min(self.dropout_p, 0.99)
        dropout_penalty_np = (p / (1.0 - p)) * np.diag(G_np)
        lambda_diag_np = dropout_penalty_np + self.reg_lambda
        
        if 'cuda' in str(self.device) and G_np.shape[0] < 20000:
            print("  inverting matrix (GPU)...")
            G_torch = torch.from_numpy(G_np).to(self.device)
            lambda_diag = torch.from_numpy(lambda_diag_np).to(self.device).float()
            del G_np, dropout_penalty_np, lambda_diag_np
            gc.collect()

            A = G_torch.clone()
            A.diagonal().add_(lambda_diag)
            P = torch.linalg.inv(A)
            del A
            
            diag_P = torch.diagonal(P)
            total_penalty = torch.maximum(lambda_diag, (1.0 - self.b) / (diag_P + self.eps))
            
            self.weight_matrix = - (P * total_penalty.unsqueeze(0))
            self.weight_matrix.diagonal().add_(1.0)
            del P
        else:
            print("  inverting matrix (CPU)...")
            # G_np is already a fresh copy
            A = G_np
            A[np.diag_indices_from(A)] += lambda_diag_np
            
            P = np.linalg.inv(A)
            del A
            
            diag_P = np.diag(P)
            total_penalty = np.maximum(lambda_diag_np, (1.0 - self.b) / (diag_P + self.eps))
            
            W = - (P * total_penalty[np.newaxis, :])
            W[np.diag_indices_from(W)] += 1.0
            
            self.weight_matrix = torch.tensor(W, dtype=torch.float32, device=self.device)
            del P, W
            
        gc.collect()
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        print("RDLAE fitting complete.")

    def forward(self, user_indices):
        """Memory-efficient hybrid inference"""
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
