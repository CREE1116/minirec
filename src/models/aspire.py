import torch
import torch.nn as nn
from .base import BaseModel

class Aspire(BaseModel):
    """
    Degree-Ratio Symmetric Wiener Filter.
    Influence I = d^2 / S, G_tilde = I^(-alpha/2) * G * I^(-alpha/2)
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.alpha = config['model'].get('alpha', 0.5)
        self.eps = 1e-12
        self.weight_matrix = None
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting Aspire (alpha={self.alpha}, lambda={self.reg_lambda}) on {self.device}...")
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X

        G = torch.sparse.mm(X.t(), X.to_dense()).to(self.device)
        d = G.diagonal()
        influence = (d ** 2) / (G.sum(dim=1) + self.eps)
        d_inv = torch.pow(influence + self.eps, -self.alpha / 2.0)
        G_tilde = G * d_inv.unsqueeze(1) * d_inv.unsqueeze(0)

        A = G_tilde.clone()
        A.diagonal().add_(self.reg_lambda)
        self.weight_matrix = torch.linalg.solve(A, G_tilde)
        print("Aspire fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)
        return self.train_matrix_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
