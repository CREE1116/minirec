import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel

class IPSWiener(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.alpha = config['model'].get('alpha', 0.5) # Implicit IPS & MNAR control
        self.eps = 1e-8
        
        self.weight_matrix = None
        self.train_matrix = sp.csr_matrix((self.n_users, self.n_items))

    def fit(self, data_loader):
        print(f"Fitting IPSWiener (alpha={self.alpha}, lambda={self.reg_lambda})...")
        train_df = data_loader.train_df
        rows, cols = train_df['user_id'].values, train_df['item_id'].values
        X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), 
                          shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix = X
        
        # Step 1: Raw Gram matrix
        G = (X.T @ X).toarray()
        
        # Step 2: Implicit IPS via Symmetric Normalization
        d = np.diag(G)
        d_inv = 1.0 / (np.power(d, self.alpha) + self.eps)
        G_tilde = G * d_inv[:, np.newaxis] * d_inv[np.newaxis, :]
        
        # Step 3: Wiener Filter (Pure Ridge Regularization, No Dropout)
        # Solve (G_tilde + lambda * I) B = G_tilde
        A = G_tilde.copy()
        np.fill_diagonal(A, np.diag(A) + self.reg_lambda)
        
        B = np.linalg.solve(A, G_tilde)
        
        self.weight_matrix = torch.from_numpy(B).float().to(self.device)
        print("IPSWiener fitting complete.")

    def forward(self, user_indices):
        u_ids = user_indices.cpu().numpy()
        user_vec = torch.from_numpy(self.train_matrix[u_ids].toarray()).float().to(self.device)
        return user_vec @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
