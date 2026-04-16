import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy


class TurboCF(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.alpha       = config['model'].get('alpha',       0.5)
        self.filter_type = config['model'].get('filter_type',   3)
        self.eps         = 1e-12

    def fit(self, data_loader):
        print(f"Fitting TurboCF (alpha={self.alpha}, filter_type={self.filter_type})...")

        X_sp = get_train_matrix_scipy(data_loader) # (U, I) sparse

        # ── 1. Asymmetric normalization: R_bar = D_U^{-alpha} X D_I^{-(1-alpha)} ──
        rowsum = np.asarray(X_sp.sum(axis=1)).ravel() + self.eps
        colsum = np.asarray(X_sp.sum(axis=0)).ravel() + self.eps

        d_u = np.power(rowsum, -self.alpha)
        d_i = np.power(colsum, -(1.0 - self.alpha))

        # We don't materialize R_bar as a dense matrix yet.
        # P_bar = R_bar^T R_bar = D_I^{-(1-alpha)} X^T D_U^{-2*alpha} X D_I^{-(1-alpha)}
        print("  Computing similarity matrix P_bar...")
        D_U_2alpha_inv = sp.diags(np.power(rowsum, -2.0 * self.alpha))
        X_scaled = D_U_2alpha_inv @ X_sp
        P_bar_sp = X_sp.T @ X_scaled # (I, I) sparse

        # Apply item-side scaling
        D_I_scaling = sp.diags(d_i)
        P_bar_sp = D_I_scaling @ P_bar_sp @ D_I_scaling
        P_bar = P_bar_sp.toarray() # (I, I) dense

        # ── 2. Row-normalize P_bar → P_hat ──
        D_P   = P_bar.sum(axis=1, keepdims=True) + self.eps
        P_hat = P_bar / D_P

        # ── 3. Polynomial LPF: H(P_hat) ──
        print(f"  Computing polynomial filter (type {self.filter_type})...")
        if self.filter_type == 1:
            H = P_hat
        elif self.filter_type == 2:
            H = 2.0 * P_hat - (P_hat @ P_hat)
        else:
            P2 = P_hat @ P_hat
            H  = 3.0 * P2 - 2.0 * (P2 @ P_hat)

        # ── 4. Final Weighting ──
        # S_hat = D_U^{-alpha} X (D_I^{-(1-alpha)} H)
        # We store W = D_I^{-(1-alpha)} H
        W = d_i.reshape(-1, 1) * H

        self.weight_matrix = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.user_scaling = torch.tensor(d_u, dtype=torch.float32, device=self.device)
        self.train_matrix_gpu = self.get_train_matrix(data_loader)
        print("TurboCF fitting complete.")

    def forward(self, user_indices):
        # input_tensor is sparse select from R (binary)
        # S_hat = d_u[u] * (R[u] @ W)
        u_scale = self.user_scaling[user_indices].unsqueeze(1)
        r_u = torch.index_select(self.train_matrix_gpu, 0, user_indices).to_dense()
        return u_scale * (r_u @ self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None