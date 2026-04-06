import torch
import torch.nn as nn
from .base import BaseModel

class GF_CF(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.k = config['model'].get('k', 256)
        self.alpha = config['model'].get('alpha', 0.3)
        self.weight_matrix = None
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting GF-CF (k={self.k}, alpha={self.alpha}) on {self.device}...")
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X

        rowsum = X.sum(dim=1)
        d_inv_row = torch.where(rowsum > 0, torch.pow(rowsum, -0.5), torch.zeros_like(rowsum))
        colsum = X.sum(dim=0)
        d_inv_col = torch.where(colsum > 0, torch.pow(colsum, -0.5), torch.zeros_like(colsum))
        R_tilde = X * d_inv_row.unsqueeze(1) * d_inv_col.unsqueeze(0)

        _, _, V = torch.svd_lowrank(R_tilde, q=self.k)

        W_linear = R_tilde.t() @ R_tilde
        V_scaled = d_inv_col.unsqueeze(1) * V
        V_inv_scaled = (1.0 / (d_inv_col + 1e-12)).unsqueeze(1) * V
        self.weight_matrix = W_linear + self.alpha * (V_scaled @ V_inv_scaled.t())
        print("GF-CF fitting complete.")

    def forward(self, user_indices):
        return self.train_matrix[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
