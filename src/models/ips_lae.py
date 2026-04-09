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
        # Use Scipy for faster sum operation
        pop = np.array(X.sum(axis=0)).flatten()
        if self.wtype == 'powerlaw':
            norm_pop = pop / (np.max(pop) + 1e-12)
            p = np.pow(norm_pop, self.wbeta)
        elif self.wtype == 'logsigmoid':
            log_freqs = np.log(pop + 1)
            alpha_logit = -self.wbeta * (np.min(log_freqs) + np.max(log_freqs)) / 2
            p = 1 / (1 + np.exp(-(alpha_logit + self.wbeta * log_freqs)))
        else:
            p = np.ones_like(pop)
        return 1 / (p + 1e-12)

    def fit(self, data_loader):
        print(f"Fitting IPS_LAE (wtype={self.wtype}, wbeta={self.wbeta}, lambda={self.reg_lambda}) on {self.device}...")
        
        X = get_train_matrix_scipy(data_loader)
        self.train_matrix_scipy = X

        print("  computing gram matrix...")
        G = compute_gram_matrix(X)
        G[np.diag_indices(self.n_items)] += self.reg_lambda
        
        print("  inverting matrix...")
        P = np.linalg.inv(G)

        B = P / (-np.diag(P) + 1e-12)
        np.fill_diagonal(B, 0)
        
        # Apply propensity weighting
        inv_p = self._compute_inv_propensity(X)
        B = B * inv_p # Element-wise multiply with items
        
        self.weight_matrix = torch.tensor(B, dtype=torch.float32, device=self.device)
        print("IPS_LAE fitting complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_matrix = self.train_matrix_scipy[users].toarray()
        input_tensor = torch.tensor(input_matrix, dtype=torch.float32, device=self.device)
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
