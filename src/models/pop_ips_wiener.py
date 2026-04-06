import torch
import torch.nn as nn
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
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting PopIPSWiener (gamma={self.propensity_gamma}, lambda={self.reg_lambda}) on {self.device}...")
        
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X
        
        # 1. Item Popularity (propensity source)
        item_pop = torch.sparse.sum(X, dim=0).to_dense()
        max_pop = torch.max(item_pop)
        
        # 2. Traditional Power-law Propensity: pi = (pop / max_pop)^gamma
        propensity = torch.pow(item_pop / (max_pop + self.eps), self.propensity_gamma)
        propensity = torch.clamp(propensity, 0.01, 1.0) # Stability clip
        
        # 3. Explicit IPS Weighting: W = diag(1/pi)
        inv_prop = 1.0 / (propensity + self.eps)
        
        # G_ips = X^T * W * X
        # To compute this efficiently: X_weighted = X * sqrt(W)
        # Then G_ips = X_weighted^T * X_weighted
        # Scaling columns of X by sqrt_inv_prop
        sqrt_inv_prop = torch.sqrt(inv_prop)
        
        # G = X'X
        G = torch.sparse.mm(X.t(), X).to_dense()
        
        # G_ips = diag(sqrt_inv_prop) @ G @ diag(sqrt_inv_prop)
        G_ips = G * sqrt_inv_prop.unsqueeze(1) * sqrt_inv_prop.unsqueeze(0)
        
        # 4. Solve Wiener Filter (Ridge): (G_ips + lambda * I) B = G_ips
        A = G_ips.clone()
        A.diagonal().add_(self.reg_lambda)
        
        B = torch.linalg.solve(A, G_ips)
        
        self.weight_matrix = B
        print("PopIPSWiener fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense()
            
        user_vec = self.train_matrix_dense[user_indices]
        return user_vec @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
