import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel
from src.utils.svd import get_svd_cache

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
        # SVD 캐시가 있더라도 d_inv_col은 나중에 W_lowpass 계산에 필요하므로 계산함
        rowsum = np.array(R.sum(axis=1)).flatten()
        d_inv_row = np.where(rowsum > 0, np.power(rowsum, -0.5), 0.)
        colsum = np.array(R.sum(axis=0)).flatten()
        d_inv_col = np.where(colsum > 0, np.power(colsum, -0.5), 0.)
        R_tilde = sp.diags(d_inv_row) @ R @ sp.diags(d_inv_col)

        # 2. SVD on R_tilde (with caching)
        k_cache = self.config.get('svd_cache_k', 1000)
        svd_data = get_svd_cache(data_loader, k_max=k_cache, matrix=R_tilde, cache_id="normalized")
        
        # Truncate to requested k
        k = min(self.k, len(svd_data['s']))
        vt = svd_data['vt'][:k, :]
        V = vt.T # (N_items, K)

        # 3. Compute W = R_tilde^T * R_tilde + alpha * D_i^-0.5 * V * V^T * D_i^0.5
        W_linear = (R_tilde.T @ R_tilde).toarray()
        
        # W_lowpass = D_i^-0.5 V V^T D_i^0.5
        V_scaled = d_inv_col[:, np.newaxis] * V 
        V_inv_scaled = (1.0 / (d_inv_col + 1e-12))[:, np.newaxis] * V 
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
