import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel

class IPS_LAE(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.wbeta = config['model'].get('wbeta', 0.4)
        self.wtype = config['model'].get('wtype', 'powerlaw') # powerlaw | logsigmoid
        self.weight_matrix = None
        self.train_matrix = None

    def _compute_inv_propensity(self, X):
        pop = np.ravel(X.sum(axis=0))
        if self.wtype == 'powerlaw':
            norm_pop = pop / (np.max(pop) + 1e-12)
            p = np.power(norm_pop, self.wbeta)
        elif self.wtype == 'logsigmoid':
            log_freqs = np.log(pop + 1)
            alpha_logit = -self.wbeta * (np.min(log_freqs) + np.max(log_freqs)) / 2
            logits = alpha_logit + self.wbeta * log_freqs
            p = 1 / (1 + np.exp(-logits))
        else:
            p = np.ones_like(pop)
        return 1 / (p + 1e-12)

    def fit(self, data_loader):
        print(f"Fitting IPS_LAE (wtype={self.wtype}, wbeta={self.wbeta})...")
        train_df = data_loader.train_df
        X = sp.csr_matrix((np.ones(len(train_df)), (train_df['user_id'], train_df['item_id'])), 
                          shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix = X
        
        # 1. Base EASE Solution: P = (X^T X + lambda I)^-1
        G = (X.T @ X).toarray()
        G += np.eye(self.n_items) * self.reg_lambda
        P = np.linalg.inv(G)
        
        # 2. B = P / -diag(P)
        diag_P = np.diag(P)
        B = P / (-diag_P + 1e-12)
        np.fill_diagonal(B, 0)
        
        # 3. Propensity Weighting
        inv_p = self._compute_inv_propensity(X)
        B = B * inv_p # Column-wise scaling
        
        self.weight_matrix = torch.from_numpy(B).float().to(self.device)
        print("IPS_LAE fitting complete.")

    def forward(self, user_indices):
        u_ids = user_indices.cpu().numpy()
        user_vec = torch.from_numpy(self.train_matrix[u_ids].toarray()).float().to(self.device)
        return user_vec @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
