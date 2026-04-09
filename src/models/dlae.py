import torch
import torch.nn as nn
import numpy as np
from scipy import sparse
from .base import BaseModel

class DLAE(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.dropout_p = config['model'].get('dropout_p', 0.5)
        self.weight_matrix = None
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting DLAE (p={self.dropout_p}, lambda={self.reg_lambda}) on {self.device}...")
        
        # Use Scipy for efficient sparse matrix multiplication
        train_df = data_loader.train_df
        row = train_df['user_id'].values
        col = train_df['item_id'].values
        data = np.ones(len(train_df), dtype=np.float32)
        X = sparse.csr_matrix((data, (row, col)), shape=(self.n_users, self.n_items))
        self.train_matrix_scipy = X

        print("  computing gram matrix...")
        G = X.T.dot(X).toarray()
        
        p = min(self.dropout_p, 0.99)
        w = (p / (1.0 - p)) * np.diag(G)

        G_lhs = G.copy()
        G_lhs[np.diag_indices(self.n_items)] += (w + self.reg_lambda)
        
        print("  solving linear system...")
        # Solving GX = B -> weight_matrix
        self.weight_matrix = torch.linalg.solve(
            torch.tensor(G_lhs, dtype=torch.float32, device=self.device),
            torch.tensor(G, dtype=torch.float32, device=self.device)
        )
        print("DLAE fitting complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_matrix = self.train_matrix_scipy[users].toarray()
        input_tensor = torch.tensor(input_matrix, dtype=torch.float32, device=self.device)
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
