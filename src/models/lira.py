import torch
import torch.nn as nn
from .base import BaseModel

class LIRA(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.weight_matrix = None # Previously self.S
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting LIRA (lambda={self.reg_lambda}) on {self.device}...")
        
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X
        
        # 1. P = (X^T X + lambda I)^-1
        G = torch.sparse.mm(X.t(), X).to_dense()
        G.diagonal().add_(self.reg_lambda)
        P = torch.linalg.inv(G)
        
        # 2. S = I - lambda * P
        # LIRA weight: S = I - lambda * (X^T X + lambda I)^-1
        S = -self.reg_lambda * P
        S.diagonal().add_(1.0)
        
        self.weight_matrix = S
        print("LIRA fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense()
            
        user_vec = self.train_matrix_dense[user_indices]
        return user_vec @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
