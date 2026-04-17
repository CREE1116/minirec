import torch
import numpy as np
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
        print(f"Fitting Adaptive ASPIRE on CPU (Simple Style)...")

        # 1. Get training data
        X_sp = get_train_matrix_scipy(data_loader) 
        self.train_matrix_cpu = X_sp.tocsr() 
        U, I = X_sp.shape

        # 2. Compute Statistics (float32)
        d_u = np.asarray(X_sp.sum(axis=1)).ravel().astype(np.float32)
        d_i = np.asarray(X_sp.sum(axis=0)).ravel().astype(np.float32)
        
        # global_dbar
        global_dbar = np.mean(d_i).astype(np.float32)
        d_bar_u_raw = (np.asarray(X_sp.dot(d_i)).ravel() / (d_u + self.eps)).astype(np.float32)
        
        k = np.float32(5.0)
        d_bar_u = ((d_u * d_bar_u_raw + k * global_dbar) / (d_u + k + self.eps)).astype(np.float32)
        
        # 3. Adaptive Gamma formula
        log_du = np.log(d_u + np.float32(1.0) + self.eps).astype(np.float32)
        log_dbar = np.log(d_bar_u + np.float32(1.0) + self.eps).astype(np.float32)
        gamma_u_star = ((log_dbar - log_du) / (log_dbar + self.eps)).astype(np.float32)
        
        # Safe Shrinkage
        gamma_u = (np.float32(0.8) * np.sqrt(np.maximum(0, gamma_u_star))).astype(np.float32)
        gamma_u = np.clip(gamma_u, 0.0, 0.9).astype(np.float32)
        
        # Pre-calculate user weights
        u_weights = np.power(d_u + self.eps, -gamma_u).astype(np.float32)
        del d_u, d_bar_u, d_bar_u_raw, log_du, log_dbar, gamma_u_star, gamma_u
        gc.collect()

        # 4. Gram Matrix (Direct Simple)
        print("  computing gram matrix...")
        # G = (X.T @ diag(w) @ X)
        # item_weights for compute_gram_matrix is D_i, result is D_i G D_i
        i_weights = np.power(d_i + self.eps, -0.5).astype(np.float32)
        G_np = compute_gram_matrix(X_sp, data_loader, weights=u_weights, item_weights=i_weights)
        
        del u_weights, i_weights, d_i
        gc.collect()

        # 5. Inversion (NumPy inv)
        print("  inverting matrix (NumPy)...")
        G_np[np.diag_indices_from(G_np)] += self.reg_lambda
        P_np = np.linalg.inv(G_np).astype(np.float32)
        del G_np
        gc.collect()

        # 6. Weights
        diag_P = np.diag(P_np).astype(np.float32)
        W_np = (-P_np / (diag_P[np.newaxis, :] + self.eps)).astype(np.float32)
        np.fill_diagonal(W_np, 0)
        del P_np, diag_P
        
        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        del W_np
        gc.collect()

        if 'cuda' in str(self.device): torch.cuda.empty_cache()
        print("Adaptive ASPIRE fitting complete.")

    def forward(self, user_indices):
        return self._get_batch_ratings(user_indices, self.weight_matrix)
