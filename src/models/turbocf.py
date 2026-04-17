import torch
import numpy as np
import scipy.sparse as sp
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix


class TurboCF(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.alpha       = np.float32(config['model'].get('alpha', 0.5))
        self.filter_type = config['model'].get('filter_type', 3)
        self.eps         = np.float32(1e-12)

    def fit(self, data_loader):
        print(f"Fitting TurboCF (alpha={self.alpha}, filter_type={self.filter_type}) on CPU float32...")

        X_sp = get_train_matrix_scipy(data_loader) # (U, I) sparse
        self.train_matrix_cpu = X_sp.tocsr() # Store for hybrid inference

        # 1. Asymmetric normalization components (float32)
        rowsum = np.asarray(X_sp.sum(axis=1)).ravel().astype(np.float32) + self.eps
        colsum = np.asarray(X_sp.sum(axis=0)).ravel().astype(np.float32) + self.eps

        d_u = np.power(rowsum, -self.alpha).astype(np.float32)
        d_i = np.power(colsum, -(np.float32(1.0) - self.alpha)).astype(np.float32)

        # 2. Optimized Block-wise Gram Matrix Construction
        print("  Computing similarity matrix P_bar (Block-wise)...")
        # G = D_i @ (X.T @ D_u^{-2*alpha} @ X) @ D_i
        user_weights = np.power(rowsum, -np.float32(2.0) * self.alpha).astype(np.float32)
        P_bar = compute_gram_matrix(X_sp, data_loader, weights=user_weights, item_weights=d_i)

        # 3. Row-normalize P_bar → P_hat
        D_P   = P_bar.sum(axis=1, keepdims=True) + self.eps
        P_hat = (P_bar / D_P).astype(np.float32)
        del P_bar
        gc.collect()

        # 4. Polynomial LPF: H(P_hat)
        print(f"  Computing polynomial filter (type {self.filter_type}) on CPU...")
        if self.filter_type == 1:
            H = P_hat
        elif self.filter_type == 2:
            # H = 2P - P^2
            P2 = (P_hat @ P_hat).astype(np.float32)
            H  = (np.float32(2.0) * P_hat - P2).astype(np.float32)
            del P2
        else:
            # H = 3P^2 - 2P^3
            P2 = (P_hat @ P_hat).astype(np.float32)
            P3 = (P2 @ P_hat).astype(np.float32)
            H  = (np.float32(3.0) * P2 - np.float32(2.0) * P3).astype(np.float32)
            del P2, P3

        # 5. Final Weighting
        W = (d_i.reshape(-1, 1) * H).astype(np.float32)

        self.weight_matrix = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.user_scaling = torch.tensor(d_u, dtype=torch.float32, device=self.device)
        
        del P_hat, H, W, d_u, d_i, rowsum, colsum
        gc.collect()
        
        print("TurboCF fitting complete.")

    def forward(self, user_indices):
        u_scale = self.user_scaling[user_indices].unsqueeze(1)
        # Hybrid inference on GPU
        scores = self._get_batch_ratings(user_indices, self.weight_matrix)
        return u_scale * scores

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
