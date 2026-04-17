import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy
from src.utils.svd import get_svd_cache

class GF_CF(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.k     = config['model'].get('k',     256)
        self.alpha = config['model'].get('alpha', 0.3)
        self.eps   = 1e-12

    def fit(self, data_loader):
        print(f"Fitting GF-CF (k={self.k}, alpha={self.alpha})...")

        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr() # Store for hybrid inference

        # 1. Symmetric normalization components
        rowsum = np.asarray(X_sp.sum(axis=1)).ravel() + self.eps
        colsum = np.asarray(X_sp.sum(axis=0)).ravel() + self.eps
        
        d_u = np.power(rowsum, -0.5)
        d_i = np.power(colsum, -0.5)
        
        # 2. Linear LPF
        print("  Computing linear filter G...")
        X_u_scaled = sp.diags(1.0 / rowsum) @ X_sp
        G_sp = X_sp.T @ X_u_scaled
        D_I_inv_half = sp.diags(d_i)
        G = (D_I_inv_half @ G_sp @ D_I_inv_half).toarray()

        # 3. Ideal LPF (SVD)
        print("  Computing ideal filter (SVD)...")
        R_tilde_sp = sp.diags(d_u) @ X_sp @ D_I_inv_half
        svd_res = get_svd_cache(data_loader, k_max=self.k, matrix=R_tilde_sp, cache_id="normalized")
        V = svd_res['vt'].T
        S_ideal = V @ V.T

        # 4. Weight matrix
        W = d_i.reshape(-1, 1) * (G + self.alpha * S_ideal)
        
        self.weight_matrix = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.user_scaling = torch.tensor(d_u, dtype=torch.float32, device=self.device)
        print("GF-CF fitting complete.")

    def forward(self, user_indices):
        u_scale = self.user_scaling[user_indices].unsqueeze(1)
        # Sparse-Dense Multiplication on GPU
        scores = self._get_batch_ratings(user_indices, self.weight_matrix)
        return u_scale * scores

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
