import torch
import torch.nn as nn
from .base import BaseModel

class PopIPSWiener(BaseModel):
    """
    Traditional IPS Wiener Filter using explicit item popularity.
    G_ips = diag(1/pi)^0.5 * G * diag(1/pi)^0.5, pi_i = (pop_i / max_pop)^gamma
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

        item_pop = torch.sparse.sum(X, dim=0).to_dense()
        propensity = torch.pow(item_pop / (torch.max(item_pop) + self.eps), self.propensity_gamma)
        propensity = torch.clamp(propensity, 0.01, 1.0)
        sqrt_inv_prop = torch.sqrt(1.0 / (propensity + self.eps)).to(self.device)

        G = torch.sparse.mm(X.t(), X.to_dense()).to(self.device)
        G_ips = G * sqrt_inv_prop.unsqueeze(1) * sqrt_inv_prop.unsqueeze(0)

        A = G_ips.clone()
        A.diagonal().add_(self.reg_lambda)
        self.weight_matrix = torch.linalg.solve(A, G_ips)
        print("PopIPSWiener fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)
        return self.train_matrix_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
