import torch
import numpy as np
import scipy.sparse as sp
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix
from src.utils.svd import get_svd_cache

class GF_CF(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.k     = config['model'].get('k', 256)
        self.alpha = np.float32(config['model'].get('alpha', 0.3))
        self.eps   = np.float32(1e-12)

    def fit(self, data_loader):
        print(f"Fitting GF-CF (k={self.k}, alpha={self.alpha}) on CPU with float32...")

        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr() # Store for hybrid inference

        # 1. Symmetric normalization components (float32)
        rowsum = np.asarray(X_sp.sum(axis=1)).ravel().astype(np.float32) + self.eps
        colsum = np.asarray(X_sp.sum(axis=0)).ravel().astype(np.float32) + self.eps
        
        d_u = np.power(rowsum, -0.5).astype(np.float32)
        d_i = np.power(colsum, -0.5).astype(np.float32)
        
        # 2. Linear LPF (Using our optimized Block-wise Gram)
        print("  Computing linear filter G (Block-wise)...")
        # G = D_i^{-0.5} @ (X.T @ D_u^{-1} @ X) @ D_i^{-0.5}
        # user_weights = 1.0 / rowsum
        user_weights = (np.float32(1.0) / rowsum).astype(np.float32)
        G = compute_gram_matrix(X_sp, data_loader, weights=user_weights, item_weights=d_i)

        # 3. Ideal LPF (SVD)
        print("  Computing ideal filter (SVD)...")
        R_tilde_sp = sp.diags(d_u, dtype=np.float32) @ X_sp @ sp.diags(d_i, dtype=np.float32)
        svd_res = get_svd_cache(data_loader, k_max=self.k, matrix=R_tilde_sp, cache_id="normalized")
        V = svd_res['vt'].T.astype(np.float32)
        
        # S_ideal = V @ V.T
        print("  Constructing S_ideal (float32)...")
        S_ideal = (V @ V.T).astype(np.float32)
        del V
        gc.collect()

        # 4. Weight matrix
        print("  Finalizing weight matrix...")
        W = (d_i.reshape(-1, 1) * (G + self.alpha * S_ideal)).astype(np.float32)
        
        self.weight_matrix = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.user_scaling = torch.tensor(d_u, dtype=torch.float32, device=self.device)
        
        del G, S_ideal, W, rowsum, colsum, d_u, d_i
        gc.collect()
        
        print("GF-CF fitting complete.")

    def forward(self, user_indices):
        u_scale = self.user_scaling[user_indices].unsqueeze(1)
        # Hybrid inference on GPU
        scores = self._get_batch_ratings(user_indices, self.weight_matrix)
        return u_scale * scores

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
