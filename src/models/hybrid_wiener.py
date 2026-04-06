import torch
import torch.nn as nn
from .base import BaseModel

class HybridWiener(BaseModel):
    """
    Hybrid 1st & 2nd order statistic Wiener Filter.
    s = (1 - gamma) * diag(G) + gamma * RowEnergy(G), G_tilde = s^-alpha * G * s^-alpha
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.alpha = config['model'].get('alpha', 0.5)
        self.gamma = config['model'].get('gamma', 0.5)
        self.eps = 1e-12
        self.weight_matrix = None
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting HybridWiener (alpha={self.alpha}, gamma={self.gamma}, lambda={self.reg_lambda}) on {self.device}...")
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X

        G = X.t() @ X
        d = G.diagonal()
        e = torch.sqrt(torch.sum(torch.square(G), dim=1))
        s = (1.0 - self.gamma) * d + self.gamma * e
        s_inv = 1.0 / (torch.pow(s + self.eps, self.alpha))
        G_tilde = G * s_inv.unsqueeze(1) * s_inv.unsqueeze(0)

        A = G_tilde.clone()
        A.diagonal().add_(self.reg_lambda)
        self.weight_matrix = torch.linalg.solve(A, G_tilde)
        print("HybridWiener fitting complete.")

    def forward(self, user_indices):
        return self.train_matrix[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
