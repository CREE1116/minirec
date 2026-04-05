import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from .base import BaseModel

class EASE(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.weight_matrix = None
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting EASE (lambda={self.reg_lambda})...")
        train_df = data_loader.train_df
        rows, cols = train_df['user_id'].values, train_df['item_id'].values
        X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix = X
        
        # G = X'X + lambda*I
        G = (X.T @ X).toarray()
        G += np.eye(self.n_items) * self.reg_lambda
        
        # P = inv(G)
        P = np.linalg.inv(G)
        
        # B = I - P / diag(P)
        B = P / (-np.diag(P))
        np.fill_diagonal(B, 0)
        
        self.weight_matrix = torch.from_numpy(B).float().to(self.device)
        print("EASE fitting complete.")

    def forward(self, user_indices):
        # user_indices: torch.LongTensor
        u_ids = user_indices.cpu().numpy()
        user_vec = torch.from_numpy(self.train_matrix[u_ids].toarray()).float().to(self.device)
        return user_vec @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
