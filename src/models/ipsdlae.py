import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel

class IPSDLAE(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.dropout_p = config['model'].get('dropout_p', 0.5)
        self.alpha = config['model'].get('alpha', 0.5) # Implicit IPS & MNAR control
        self.eps = 1e-8
        
        self.weight_matrix = None
        self.train_matrix = sp.csr_matrix((self.n_users, self.n_items))

    def fit(self, data_loader):
        print(f"Fitting IPSDLAE (Implicit IPS via Normalization, alpha={self.alpha})...")
        train_df = data_loader.train_df
        rows, cols = train_df['user_id'].values, train_df['item_id'].values
        X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), 
                          shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix = X
        
        # Step 1
        G = (X.T @ X).toarray()

        # Step 2 (MNAR / IPS implicit)
        d = np.diag(G)
        d_inv = 1.0 / (np.power(d, self.alpha) + self.eps)
        G_tilde = G * d_inv[:, None] * d_inv[None, :]

        # Step 3 (DLAE - FIXED)
        g_diag = np.diag(G)   # ← 핵심 수정
        p = min(self.dropout_p, 0.99)
        w = (p / (1.0 - p + self.eps)) * g_diag
        L_diag = w + self.reg_lambda

        # Step 4
        A = G_tilde + np.diag(L_diag)
        B = np.linalg.solve(A, G_tilde)
        
        self.weight_matrix = torch.from_numpy(B).float().to(self.device)
        print("IPSDLAE fitting complete.")

    def forward(self, user_indices):
        u_ids = user_indices.cpu().numpy()
        user_vec = torch.from_numpy(self.train_matrix[u_ids].toarray()).float().to(self.device)
        return user_vec @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
