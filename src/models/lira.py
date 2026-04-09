import torch
import torch.nn as nn
import numpy as np
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class LIRA(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.weight_matrix = None
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting LIRA (lambda={self.reg_lambda}) on {self.device}...")
        
        X = get_train_matrix_scipy(data_loader)
        self.train_matrix_scipy = X

        print("  computing gram matrix...")
        G = compute_gram_matrix(X)
        G[np.diag_indices(self.n_items)] += self.reg_lambda
        
        print("  inverting matrix...")
        P = np.linalg.inv(G)

        # S = I - lambda * (G + lambda*I)^-1
        S = -self.reg_lambda * P
        np.fill_diagonal(S, S.diagonal() + 1.0)
        
        self.weight_matrix = torch.tensor(S, dtype=torch.float32, device=self.device)
        print("LIRA fitting complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_matrix = self.train_matrix_scipy[users].toarray()
        input_tensor = torch.tensor(input_matrix, dtype=torch.float32, device=self.device)
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
