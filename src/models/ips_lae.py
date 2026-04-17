import torch
import numpy as np
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class IPS_LAE(BaseModel):
    """
    IPS_LAE: Inverse Propensity Scored Linear AutoEncoder
    Optimized with NumPy inv and Strict float32.
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = np.float32(config['model'].get('reg_lambda', 500.0))
        self.wbeta = np.float32(config['model'].get('wbeta', 0.4))
        self.wtype = config['model'].get('wtype', 'logsigmoid')
        self.eps = np.float32(1e-12)
        self.weight_matrix = None

    def _compute_inv_propensity(self, X):
        pop = np.array(X.sum(axis=0)).flatten().astype(np.float32)
        if self.wtype == 'powerlaw':
            norm_pop = (pop / (np.max(pop) + self.eps)).astype(np.float32)
            p = np.power(norm_pop, self.wbeta).astype(np.float32)
        elif self.wtype == 'logsigmoid':
            log_freqs = np.log(pop + np.float32(1.0)).astype(np.float32)
            alpha_logit = -self.wbeta * (np.min(log_freqs) + np.max(log_freqs)) / np.float32(2.0)
            p = (np.float32(1.0) / (np.float32(1.0) + np.exp(-(alpha_logit + self.wbeta * log_freqs)))).astype(np.float32)
        else:
            p = np.ones_like(pop, dtype=np.float32)
        
        inv_p = (np.float32(1.0) / (p + self.eps)).astype(np.float32)
        return torch.tensor(inv_p, dtype=torch.float32, device=self.device)

    def fit(self, data_loader):
        print(f"Fitting IPS_LAE (lambda={self.reg_lambda}) on CPU using NumPy inv...")
        
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr()

        print("  computing gram matrix (CPU float32)...")
        G_np = compute_gram_matrix(X_sp, data_loader)
        gc.collect()
        
        print("  inverting matrix (NumPy float32)...")
        G_np[np.diag_indices_from(G_np)] += self.reg_lambda
        
        P_np = np.linalg.inv(G_np).astype(np.float32)
        del G_np 
        gc.collect()
        
        diag_P = np.diag(P_np).astype(np.float32)
        B_np = (-P_np / (diag_P + self.eps)).astype(np.float32)
        np.fill_diagonal(B_np, 0)
        del P_np, diag_P
        gc.collect()
        
        # Move to GPU
        B_gpu = torch.tensor(B_np, dtype=torch.float32, device=self.device)
        del B_np 
        gc.collect()
        
        # Apply IPS weighting
        inv_p = self._compute_inv_propensity(X_sp)
        self.weight_matrix = B_gpu * inv_p.view(1, -1)
        self.weight_matrix.diagonal().zero_()
        
        del B_gpu, inv_p
        gc.collect()
        
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        print("IPS_LAE fitting complete.")

    def forward(self, user_indices):
        """Memory-efficient hybrid inference"""
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
