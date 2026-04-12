import torch
import torch.nn as nn
import numpy as np
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class IPS_LAE(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.wbeta = config['model'].get('wbeta', 0.4)
        self.wtype = config['model'].get('wtype', 'logsigmoid')  # powerlaw | logsigmoid
        self.weight_matrix = None
        self.train_matrix_scipy = None

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
        print(f"Fitting IPS_LAE (lambda={self.reg_lambda}) on {self.device}...")
        
        X = get_train_matrix_scipy(data_loader)
        self.train_matrix_scipy = X

        print("  computing gram matrix (CPU)...")
        G_np = compute_gram_matrix(X, data_loader)
        
        print("  inverting matrix (CPU)...")
        # G = G + lambda * I
        G_np[np.diag_indices_from(G_np)] += self.reg_lambda
        
        P_np = np.linalg.inv(G_np)
        
        # Final weights: B_{ij} = -P_{ij} / P_{jj} (i != j), B_{ii} = 0
        diag_P = np.diag(P_np)
        B_np = P_np / (-diag_P[:, np.newaxis] + 1e-12)
        np.fill_diagonal(B_np, 0)
        
        B = torch.tensor(B_np, dtype=torch.float32, device=self.device)
        
        # Apply propensity weighting on GPU
        inv_p = self._compute_inv_propensity(X)
        B = B * inv_p.view(1, -1)
        
        self.weight_matrix = B
        print("IPS_LAE fitting complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_matrix = self.train_matrix_scipy[users].toarray()
        input_tensor = torch.tensor(input_matrix, dtype=torch.float32, device=self.device)
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
