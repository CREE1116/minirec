import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel

class PopIPSWiener(BaseModel):
    """
    Traditional IPS-based Wiener Filter using Explicit Item Popularity.
    G_ips = X^T * diag(1/pi) * X
    where pi_i = (pop_i / max_pop)^gamma
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.propensity_gamma = config['model'].get('propensity_gamma', 0.5)
        self.eps = 1e-8
        
        self.weight_matrix = None
        self.train_matrix = sp.csr_matrix((self.n_users, self.n_items))

    def fit(self, data_loader):
        print(f"Fitting PopIPSWiener (gamma={self.propensity_gamma}, lambda={self.reg_lambda})...")
        train_df = data_loader.train_df
        rows, cols = train_df['user_id'].values, train_df['item_id'].values
        X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), 
                          shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix = X
        
        # 1. Item Popularity (propensity source)
        item_pop = np.array(X.sum(axis=0)).flatten()
        max_pop = np.max(item_pop)
        
        # 2. Traditional Power-law Propensity: pi = (pop / max_pop)^gamma
        propensity = np.power(item_pop / (max_pop + self.eps), self.propensity_gamma)
        propensity = np.clip(propensity, 0.01, 1.0) # Stability clip
        
        # 3. Explicit IPS Weighting: W = diag(1/pi)
        inv_prop = 1.0 / (propensity + self.eps)
        
        # G_ips = X^T * W * X
        # To compute this efficiently: X_weighted = X * sqrt(W)
        # Then G_ips = X_weighted^T * X_weighted
        sqrt_inv_prop = np.sqrt(inv_prop)
        X_weighted = X.multiply(sqrt_inv_prop)
        
        G_ips = (X_weighted.T @ X_weighted).toarray()
        
        # 4. Solve Wiener Filter (Ridge): (G_ips + lambda * I) B = G_ips
        A = G_ips.copy()
        np.fill_diagonal(A, np.diag(A) + self.reg_lambda)
        
        B = np.linalg.solve(A, G_ips)
        
        self.weight_matrix = torch.from_numpy(B).float().to(self.device)
        print("PopIPSWiener fitting complete.")

    def forward(self, user_indices):
        u_ids = user_indices.cpu().numpy()
        user_vec = torch.from_numpy(self.train_matrix[u_ids].toarray()).float().to(self.device)
        return user_vec @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
