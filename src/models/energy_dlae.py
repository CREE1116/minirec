import torch
import torch.nn as nn
from .base import BaseModel

class EnergyDLAE(BaseModel):
    """
    Energy-based DLAE: combines DLAE's dropout regularization with 2nd-order energy normalization.
    G_tilde = E^-alpha * G * E^-alpha, W = (G_tilde + diag(p/(1-p)*g_tilde + lambda))^-1 * G_tilde
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.dropout_p = config['model'].get('dropout_p', 0.5)
        self.alpha = config['model'].get('alpha', 0.5)
        self.eps = 1e-12
        self.weight_matrix = None
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting EnergyDLAE (p={self.dropout_p}, alpha={self.alpha}, lambda={self.reg_lambda}) on {self.device}...")
        self.train_matrix = self.get_train_matrix(data_loader)
        X = self.train_matrix

        G = torch.sparse.mm(X.t(), X.to_dense()).to(self.device)

        row_energy = torch.sqrt(torch.sum(torch.square(G), dim=1))
        e_inv = 1.0 / (torch.pow(row_energy + self.eps, self.alpha))
        G_tilde = G * e_inv.unsqueeze(1) * e_inv.unsqueeze(0)

        g_tilde_diag = G_tilde.diagonal()
        p = min(self.dropout_p, 0.99)
        w = (p / (1.0 - p + self.eps)) * g_tilde_diag

        A = G_tilde.clone()
        A.diagonal().add_(w + self.reg_lambda)
        self.weight_matrix = torch.linalg.solve(A, G_tilde)
        print("EnergyDLAE fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)
        return self.train_matrix_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
