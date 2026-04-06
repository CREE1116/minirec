import torch
import torch.nn as nn
from .base import BaseModel

class IPSWiener(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.alpha = config['model'].get('alpha', 0.5)
        self.eps = 1e-8
        self.weight_matrix = None
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting IPSWiener (alpha={self.alpha}, lambda={self.reg_lambda}) on {self.device}...")
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X

        G = torch.sparse.mm(X.t(), X.to_dense()).to(self.device)
        d = G.diagonal()
        d_inv = 1.0 / (torch.pow(d, self.alpha/2) + self.eps)
        G_tilde = G * d_inv.unsqueeze(1) * d_inv.unsqueeze(0)

        A = G_tilde.clone()
        A.diagonal().add_(self.reg_lambda)
        self.weight_matrix = torch.linalg.solve(A, G_tilde)
        print("IPSWiener fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)
        return self.train_matrix_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
