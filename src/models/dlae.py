import torch
import torch.nn as nn
from .base import BaseModel

class DLAE(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.dropout_p = config['model'].get('dropout_p', 0.5)
        self.weight_matrix = None
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting DLAE (p={self.dropout_p}, lambda={self.reg_lambda}) on {self.device}...")
        self.train_matrix = self.get_train_matrix(data_loader)
        X = self.train_matrix
        
        # Step 1: Gram matrix G = X.T @ X (I x I)
        # Using sparse.mm(X.t(), X.to_dense()) to get dense G efficiently
        G = torch.sparse.mm(X.t(), X.to_dense())
        
        # Step 2: g_diag = diag(G)
        g_diag = G.diagonal()
        
        # Step 3: w = (p / (1-p)) * g_diag
        p = self.dropout_p
        if p >= 1.0: p = 0.99 # Safety clamp
        w = (p / (1.0 - p)) * g_diag
        
        # Step 4 & 5: Solve (G + diag(w + lambda)) B = G
        G_lhs = G.clone()
        G_lhs.diagonal().add_(w + self.reg_lambda)
        
        self.weight_matrix = torch.linalg.solve(G_lhs, G)
        print("DLAE fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense()
        user_vec = self.train_matrix_dense[user_indices]
        return user_vec @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
