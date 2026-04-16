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

        X_sp = get_train_matrix_scipy(data_loader) # (U, I) sparse

        # ── 1. Symmetric normalization components ──
        rowsum = np.asarray(X_sp.sum(axis=1)).ravel() + self.eps
        colsum = np.asarray(X_sp.sum(axis=0)).ravel() + self.eps
        
        d_u = np.power(rowsum, -0.5)
        d_i = np.power(colsum, -0.5)
        
        # ── 2. Linear LPF: G = R_tilde^T R_tilde ──
        # G = D_I^{-0.5} X^T D_U^{-1} X D_I^{-0.5}
        print("  Computing linear filter G...")
        X_u_scaled = sp.diags(1.0 / rowsum) @ X_sp
        G_sp = X_sp.T @ X_u_scaled # (I, I) sparse
        
        # Apply item-side scaling to G
        D_I_inv_half = sp.diags(d_i)
        G_sp = D_I_inv_half @ G_sp @ D_I_inv_half
        G = G_sp.toarray() # (I, I) dense

        # ── 3. Ideal LPF: V_k V_k^T ──
        print("  Computing ideal filter (SVD)...")
        R_tilde_sp = sp.diags(d_u) @ X_sp @ D_I_inv_half
        svd_res = get_svd_cache(data_loader, k_max=self.k, matrix=R_tilde_sp, cache_id="normalized")
        V = svd_res['vt'].T # (I, k)
        S_ideal = V @ V.T # (I, I)

        # ── 4. Combined Weight Matrix ──
        # s_u = d_u[u] * r_u @ [D_I^{-0.5} @ (G + alpha * S_ideal)]
        W = d_i.reshape(-1, 1) * (G + self.alpha * S_ideal)
        
        self.weight_matrix = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.user_scaling = torch.tensor(d_u, dtype=torch.float32, device=self.device)
        self.train_matrix_gpu = self.get_train_matrix(data_loader)
        print("GF-CF fitting complete.")

    def forward(self, user_indices):
        u_scale = self.user_scaling[user_indices].unsqueeze(1)
        r_u = torch.index_select(self.train_matrix_gpu, 0, user_indices).to_dense()
        return u_scale * (r_u @ self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
