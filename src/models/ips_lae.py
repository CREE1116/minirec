import torch
import numpy as np
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class IPS_LAE(BaseModel):
    """
    IPS_LAE: Inverse Propensity Scored Linear AutoEncoder
    Optimized with Hybrid CPU/GPU inference.
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.wbeta = config['model'].get('wbeta', 0.4)
        self.wtype = config['model'].get('wtype', 'logsigmoid')
        self.weight_matrix = None

    def _compute_inv_propensity(self, X):
        pop = np.array(X.sum(axis=0)).flatten()
        if self.wtype == 'powerlaw':
            norm_pop = pop / (np.max(pop) + 1e-12)
            p = np.power(norm_pop, self.wbeta)
        elif self.wtype == 'logsigmoid':
            log_freqs = np.log(pop + 1)
            alpha_logit = -self.wbeta * (np.min(log_freqs) + np.max(log_freqs)) / 2
            p = 1 / (1 + np.exp(-(alpha_logit + self.wbeta * log_freqs)))
        else:
            p = np.ones_like(pop)
        return torch.tensor(1 / (p + 1e-12), dtype=torch.float32, device=self.device)

    def fit(self, data_loader):
        print(f"Fitting IPS_LAE (lambda={self.reg_lambda}) on CPU/GPU Hybrid...")
        
        # 1. Load data onto CPU
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr()

        print("  computing gram matrix (CPU)...")
        G_np = compute_gram_matrix(X_sp, data_loader)
        
        # 2. Ridge Inversion on CPU
        print("  inverting matrix (CPU)...")
        G_np[np.diag_indices_from(G_np)] += self.reg_lambda
        P_np = np.linalg.inv(G_np)
        
        diag_P = np.diag(P_np)
        B_np = -P_np / (diag_P + 1e-12)
        np.fill_diagonal(B_np, 0)
        
        B_gpu = torch.tensor(B_np, dtype=torch.float32, device=self.device)
        
        # 3. Apply IPS weighting on GPU
        inv_p = self._compute_inv_propensity(X_sp)
        self.weight_matrix = B_gpu * inv_p.view(1, -1)
        
        print("IPS_LAE fitting complete.")

    def forward(self, user_indices):
        """Memory-efficient hybrid inference"""
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
