import torch
import numpy as np
import scipy.sparse as sp
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy

class AdaptiveAspire(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.damping_coeff = config['model'].get('damping_coeff', 3.0)
        self.eps = 1e-12
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting Adaptive ASPIRE on {self.device}...")

        # 1. Get training data as Scipy CSR (Keep on CPU)
        X_sp = get_train_matrix_scipy(data_loader) 
        self.train_matrix_cpu = X_sp.tocsr() # Store for inference
        U, I = X_sp.shape

        # 2. Compute Proxies (Point 9 & New Theorem)
        d_u = np.asarray(X_sp.sum(axis=1)).ravel()
        d_i = np.asarray(X_sp.sum(axis=0)).ravel()
        
        # d_norm: user activity as probability
        p_u_proxy = (d_u + self.eps) / (I + self.eps)
        # d_bar_u: average item popularity seen by user
        # (X @ d_i) / d_u gives mean item degree
        c_u_proxy = (np.asarray(X_sp.dot(d_i)).ravel() / (d_u + self.eps)) / (U + self.eps)
        
        # 3. New Theoretical Crossover: gamma* = 2*log(1/pu) / log(cu/pu)
        log_pu_inv = -np.log(p_u_proxy + self.eps)
        log_cu_pu_ratio = np.log((c_u_proxy / (p_u_proxy + self.eps)) + self.eps)
        
        valid = (d_u > 1) & (log_cu_pu_ratio > 0)
        # Formula: gamma* = 2 * log(1/pu) / log(cu/pu)
        gamma_star_u = np.where(valid, (2 * log_pu_inv) / (log_pu_inv + np.log(c_u_proxy + self.eps)), 0.0)
        
        # 4. Statistical Noise Damping
        avg_d = np.mean(d_u)
        var_d = np.var(d_u)
        density = avg_d / I
        expected_var_random = avg_d * (1 - density)
        noise_ratio = expected_var_random / (var_d + expected_var_random + self.eps)
        sigma_est = noise_ratio * 0.5 
        
        damping_factor = max(0.1, 1.0 - self.damping_coeff * sigma_est)
        # Apply damping and ensure bound
        gamma_star_u = (gamma_star_u * damping_factor).clip(0.0, 1.2)
        
        print(f"  -> Damping Factor: {damping_factor:.4f}, Mean Gamma*: {np.mean(gamma_star_u[valid]):.4f}")
        
        # 5. Construct Adaptive Gram Matrix (CPU Sparse Multiplication)
        print("  Constructing G_tilde (CPU Sparse)...")
        user_weights = np.power(d_u + self.eps, -gamma_star_u)
        D_U_adaptive = sp.diags(user_weights)
        item_weights = np.power(d_i + self.eps, -0.5) # Item-side remains sqrt for stability
        D_I_inv_half = sp.diags(item_weights)

        G_mid = X_sp.T @ D_U_adaptive @ X_sp
        G_tilde_np = (D_I_inv_half @ G_mid @ D_I_inv_half).toarray().astype(np.float32)

        del d_u, d_i, p_u_proxy, c_u_proxy, valid, user_weights, D_U_adaptive, item_weights, D_I_inv_half, G_mid
        gc.collect()

        # 6. Solve Ridge
        if 'cuda' in str(self.device):
            print("  Solving Ridge Regression (GPU)...")
            G_torch = torch.from_numpy(G_tilde_np).to(self.device)
            del G_tilde_np
            gc.collect()

            G_torch.diagonal().add_(self.reg_lambda)
            P = torch.linalg.inv(G_torch)
            del G_torch
            
            P_diag = torch.diagonal(P)
            self.weight_matrix = -P / (P_diag.unsqueeze(0) + self.eps)
            self.weight_matrix.diagonal().zero_()
            del P
        else:
            print("  [Warning] CUDA not available, falling back to CPU...")
            G_tilde_np[np.diag_indices_from(G_tilde_np)] += self.reg_lambda
            P_np = np.linalg.inv(G_tilde_np)
            del G_tilde_np
            gc.collect()

            P_diag = np.diag(P_np)
            W_np = -P_np / (P_diag[np.newaxis, :] + self.eps)
            np.fill_diagonal(W_np, 0)
            
            self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
            del P_np, W_np

        gc.collect()
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        print("Adaptive ASPIRE fitting complete.")

    def forward(self, user_indices):
        return self._get_batch_ratings(user_indices, self.weight_matrix)
