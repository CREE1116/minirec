import torch
import numpy as np
import gc
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
        print(f"Fitting DLAE (p={self.dropout_p}, lambda={self.reg_lambda}) on {self.device}...")
        
        # 1. Load data onto CPU
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr()

        print(f"  computing gram matrix (on {self.device})...")
        G_np = compute_gram_matrix(X_sp, data_loader, device=self.device)
        
        # 2. Add Dropout Penalty: diag += (p/(1-p)) * G_jj
        p = min(self.dropout_p, 0.99)
        dropout_penalty = (p / (1.0 - p)) * np.diag(G_np)
        
        if 'cuda' in str(self.device):
            print("  solving linear system (GPU)...")
            G_rhs = torch.from_numpy(G_np).to(self.device)
            G_lhs = G_rhs.clone()
            G_lhs.diagonal().add_(torch.from_numpy(dropout_penalty).to(self.device).float() + self.reg_lambda)
            
            self.weight_matrix = torch.linalg.solve(G_lhs, G_rhs)
            
            del G_lhs, G_rhs, G_np, dropout_penalty
        else:
            print("  [Warning] CUDA not available, falling back to CPU...")
            # G_np is already a fresh copy
            G_lhs = G_np
            G_lhs[np.diag_indices_from(G_lhs)] += (dropout_penalty + self.reg_lambda)
            
            # G_orig 확보 (RHS 용)
            G_orig = compute_gram_matrix(X_sp, data_loader)
            
            W_np = np.linalg.solve(G_lhs, G_orig)
            
            self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
            del G_lhs, G_orig, W_np, dropout_penalty

        gc.collect()
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        print("DLAE fitting complete.")

    def forward(self, user_indices):
        """Memory-efficient hybrid inference"""
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
