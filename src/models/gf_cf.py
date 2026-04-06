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
        self.train_matrix = self.get_train_matrix(data_loader)
        R = self.train_matrix

        rowsum = torch.sparse.sum(R, dim=1).to_dense()
        d_inv_row = torch.where(rowsum > 0, torch.pow(rowsum, -0.5), torch.zeros_like(rowsum))
        colsum = torch.sparse.sum(R, dim=0).to_dense()
        d_inv_col = torch.where(colsum > 0, torch.pow(colsum, -0.5), torch.zeros_like(colsum))

        indices = R.indices()
        values = R.values() * d_inv_row[indices[0]] * d_inv_col[indices[1]]
        R_tilde = torch.sparse_coo_tensor(indices, values, R.shape).coalesce()

        _, _, V = torch.svd_lowrank(R_tilde, q=self.k)

        W_linear = torch.sparse.mm(R_tilde.t(), R_tilde.to_dense()).to(self.device)
        d_inv_col = d_inv_col.to(self.device)
        V = V.to(self.device)
        V_scaled = d_inv_col.unsqueeze(1) * V
        V_inv_scaled = (1.0 / (d_inv_col + 1e-12)).unsqueeze(1) * V
        W_lowpass = V_scaled @ V_inv_scaled.t()

        self.weight_matrix = W_linear + self.alpha * W_lowpass
        print("GF-CF fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)
        return self.train_matrix_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
