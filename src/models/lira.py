import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel

class LIRA(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.S = None
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting LIRA (lambda={self.reg_lambda})...")
        train_df = data_loader.train_df
        X = sp.csr_matrix((np.ones(len(train_df)), (train_df['user_id'], train_df['item_id'])), 
                          shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix = X
        
        # 1. P = (X^T X + lambda I)^-1
        G = (X.T @ X).toarray()
        G += np.eye(self.n_items) * self.reg_lambda
        P = np.linalg.inv(G)
        
        # 2. S = I - lambda * P
        # LIRA weight: S = I - lambda * (X^T X + lambda I)^-1
        S = -self.reg_lambda * P
        np.fill_diagonal(S, S.diagonal() + 1.0)
        
        self.S = torch.from_numpy(S).float().to(self.device)
        print("LIRA fitting complete.")

    def forward(self, user_indices):
        u_ids = user_indices.cpu().numpy()
        user_vec = torch.from_numpy(self.train_matrix[u_ids].toarray()).float().to(self.device)
        return user_vec @ self.S

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
