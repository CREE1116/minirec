import torch
import torch.nn as nn
import numpy as np
from scipy import sparse
from .base import BaseModel

class EASE(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.weight_matrix = None
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting EASE (lambda={self.reg_lambda}) on {self.device}...")
        
        # 1. Gram Matrix calculation on CPU (Sparse)
        train_df = data_loader.train_df
        row = train_df['user_id'].values
        col = train_df['item_id'].values
        data = np.ones(len(train_df), dtype=np.float32)
        X = sparse.csr_matrix((data, (row, col)), shape=(self.n_users, self.n_items))
        self.train_matrix_scipy = X

        print("  computing gram matrix (CPU)...")
        G = X.T.dot(X).toarray()
        
        # 2. Move to GPU for fast inversion
        print(f"  inverting matrix on {self.device}...")
        G = torch.tensor(G, dtype=torch.float32, device=self.device)
        G.diagonal().add_(self.reg_lambda)
        
        P = torch.linalg.inv(G)
        
        # 3. Final weights
        B = P / (-P.diagonal().view(-1, 1) + 1e-12)
        B.diagonal().zero_()
        
        self.weight_matrix = B
        print("EASE fitting complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_matrix = self.train_matrix_scipy[users].toarray()
        input_tensor = torch.tensor(input_matrix, dtype=torch.float32, device=self.device)
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
