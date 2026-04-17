import torch
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class AdaptiveAspire(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = np.float32(config['model'].get('reg_lambda', 100.0))
        self.eps = np.float32(1e-12)
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting Adaptive ASPIRE on CPU...")

        # 1. Get training data as Scipy CSR
        X_sp = get_train_matrix_scipy(data_loader) 
        self.train_matrix_cpu = X_sp.tocsr() 
        U, I = X_sp.shape

        # 2. Compute Degrees
        d_u = np.asarray(X_sp.sum(axis=1)).ravel().astype(np.float32)
        d_i = np.asarray(X_sp.sum(axis=0)).ravel().astype(np.float32)
        
        # 3. Compute Mean Item Popularity per User (d_bar_u) with Global Prior
        global_dbar = np.mean(d_i).astype(np.float32)
        d_bar_u_raw = (np.asarray(X_sp.dot(d_i)).ravel() / (d_u + self.eps)).astype(np.float32)
        
        k = np.float32(5.0)
        d_bar_u = ((d_u * d_bar_u_raw + k * global_dbar) / (d_u + k + self.eps)).astype(np.float32)
        
        # 4. Apply Adaptive Formula with Log-Smoothing (Force float32)
        log_du = np.log(d_u + np.float32(1.0) + self.eps).astype(np.float32)
        log_dbar = np.log(d_bar_u + np.float32(1.0) + self.eps).astype(np.float32)
        
        gamma_u_star = ((log_dbar - log_du) / (log_dbar + self.eps)).astype(np.float32)
        
        # 5. Apply Controlled Safe Shrinkage
        self.gamma_star_u = (np.float32(0.8) * np.sqrt(np.maximum(0, gamma_u_star))).astype(np.float32)
        self.gamma_star_u = np.clip(self.gamma_star_u, 0.0, 0.9).astype(np.float32)
        
        print(f"  -> Global d_bar: {global_dbar:.2f}")
        print(f"  -> Mean Adaptive Gamma (Safe & Smoothed): {np.mean(self.gamma_star_u[d_u > 0]):.4f}")
        
        # 6. Construct Adaptive Gram Matrix
        print("  Constructing G_tilde (CPU Sparse)...")
        user_weights = np.power(d_u + self.eps, -self.gamma_star_u).astype(np.float32)
        D_U_adaptive = sp.diags(user_weights, dtype=np.float32)
        item_weights = np.power(d_i + self.eps, -0.5).astype(np.float32)
        D_I_inv_half = sp.diags(item_weights, dtype=np.float32)

        G_mid = X_sp.T @ D_U_adaptive @ X_sp
        G_tilde_np = (D_I_inv_half @ G_mid @ D_I_inv_half).toarray().astype(np.float32)

        del d_u, d_i, log_du, log_dbar, gamma_u_star, user_weights, D_U_adaptive, item_weights, D_I_inv_half, G_mid
        gc.collect()

        # 7. Solve Ridge on CPU (NumPy float32)
        print("  Solving Ridge Regression (CPU In-place float32)...")
        G_tilde_np[np.diag_indices_from(G_tilde_np)] += self.reg_lambda
        P_np = la.inv(G_tilde_np, overwrite_a=True).astype(np.float32)
        del G_tilde_np
        gc.collect()

        P_diag = np.diag(P_np).astype(np.float32)
        W_np = (-P_np / (P_diag[np.newaxis, :] + self.eps)).astype(np.float32)
        np.fill_diagonal(W_np, 0)
        
        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        del P_np, W_np

        gc.collect()
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        print("Adaptive ASPIRE fitting complete.")

    def forward(self, user_indices):
        return self._get_batch_ratings(user_indices, self.weight_matrix)
