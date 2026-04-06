import torch
import torch.nn as nn
from .base import BaseModel

class LIRA(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.weight_matrix = None
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting LIRA (lambda={self.reg_lambda}) on {self.device}...")
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X

        G = X.t() @ X
        G.diagonal().add_(self.reg_lambda)
        P = torch.linalg.inv(G)

        S = -self.reg_lambda * P
        S.diagonal().add_(1.0)
        self.weight_matrix = S
        print("LIRA fitting complete.")

    def forward(self, user_indices):
        return self.train_matrix[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
