import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from .base import BaseModel

class GF_CF(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.k = config['model'].get('k', 256)
        self.alpha = config['model'].get('alpha', 0.3)
        self.weight_matrix = None
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting GF-CF (k={self.k}, alpha={self.alpha})...")
        train_df = data_loader.train_df
        R = sp.csr_matrix((np.ones(len(train_df)), (train_df['user_id'], train_df['item_id'])), 
                          shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix = R

        # 1. Normalization (D^-0.5 * R * D^-0.5)
        rowsum = np.array(R.sum(axis=1)).flatten()
        d_inv_row = np.where(rowsum > 0, np.power(rowsum, -0.5), 0.)
        colsum = np.array(R.sum(axis=0)).flatten()
        d_inv_col = np.where(colsum > 0, np.power(colsum, -0.5), 0.)
        R_tilde = sp.diags(d_inv_row) @ R @ sp.diags(d_inv_col)

        # 2. SVD on R_tilde
        k = min(self.k, min(R_tilde.shape) - 1)
        _, s, vt = svds(R_tilde, k=k)
        
        # Sort descending
        idx = np.argsort(s)[::-1]
        s, vt = s[idx], vt[idx, :]
        V = vt.T # (N_items, K)

        # 3. Compute W = R_tilde^T * R_tilde + alpha * D_i^-0.5 * V * V^T * D_i^0.5
        # For simplicity and small-to-mid scale datasets, we materialize W.
        # W_linear = R_tilde^T * R_tilde
        W_linear = (R_tilde.T @ R_tilde).toarray()
        
        # W_lowpass = V * V^T with degree scale
        # L = D_i^-0.5 V V^T D_i^0.5
        V_scaled = d_inv_col[:, np.newaxis] * V # D_i^-0.5 * V
        V_inv_scaled = (1.0 / (d_inv_col + 1e-12))[:, np.newaxis] * V # D_i^0.5 * V
        W_lowpass = V_scaled @ V_inv_scaled.T
        
        W = W_linear + self.alpha * W_lowpass
        self.weight_matrix = torch.from_numpy(W).float().to(self.device)
        print("GF-CF fitting complete.")

    def forward(self, user_indices):
        u_ids = user_indices.cpu().numpy()
        user_vec = torch.from_numpy(self.train_matrix[u_ids].toarray()).float().to(self.device)
        return user_vec @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
