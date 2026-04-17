import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy

class AdaptiveAspire(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.damping_coeff = config['model'].get('damping_coeff', 3.0)
        self.eps = 1e-12
        self.weight_matrix = None
        self.train_matrix_gpu = None

    def fit(self, data_loader):
        print("Fitting Adaptive ASPIRE (Statistical Noise Estimation)...")

        # 1. Get training data as Scipy CSR
        X_sp = get_train_matrix_scipy(data_loader) # (U, I) sparse
        U, I = X_sp.shape

        # 2. Compute Proxies
        d_u = np.asarray(X_sp.sum(axis=1)).ravel()
        d_i = np.asarray(X_sp.sum(axis=0)).ravel()
        
        # 3. Calculate Adaptive Gamma*
        p_u_proxy = (d_u + self.eps) / (d_u.max() + self.eps)
        d_bar_u = np.asarray(X_sp.dot(d_i)).ravel() / (d_u + self.eps)
        
        valid = (d_u > 1) & (d_bar_u > 1)
        gamma_star_u = np.where(valid, -np.log(p_u_proxy) / np.log(d_bar_u + self.eps), 0.0)
        
        # --- NEW: Statistical Noise Estimation (No SVD) ---
        # Principle: In a pure random graph (noise), Var(d_u) == mean(d_u) * (1 - density).
        # In a structured graph (signal), Var(d_u) >> Random_Var.
        # We estimate sigma by the ratio of 'Randomness' in the degree distribution.
        
        print("  Estimating noise level (sigma) using degree dispersion statistics...")
        avg_d = np.mean(d_u)
        var_d = np.var(d_u)
        density = avg_d / I
        
        # Theoretical variance of a random (Poisson/Binomial) degree distribution
        expected_var_random = avg_d * (1 - density)
        
        # Noise Proxy: How much of our degree variance looks like random noise?
        # If var_d is close to expected_var_random, it's pure noise (noise_ratio -> 1)
        noise_ratio = expected_var_random / (var_d + expected_var_random + self.eps)
        
        # Map this ratio to a sigma-like scale (0.0 to 0.3)
        # This is a heuristic that performs similarly to the SVD residual std.
        sigma_est = noise_ratio * 0.5 
        
        damping_factor = max(0.1, 1.0 - self.damping_coeff * sigma_est)
        
        print(f"  -> Avg Degree: {avg_d:.2f}, Var Degree: {var_d:.2f}")
        print(f"  -> Statistical Sigma Proxy: {sigma_est:.4f}")
        print(f"  -> Damping Factor: {damping_factor:.4f}")
        
        # Apply the dynamically computed scaling factor
        gamma_star_u = gamma_star_u * damping_factor
        gamma_star_u = np.clip(gamma_star_u, 0.0, 1.0)

        print(f"  Mean Adaptive Gamma: {np.mean(gamma_star_u):.4f}")
        
        # 4. Construct Adaptive Gram Matrix
        user_weights = np.power(d_u + self.eps, -gamma_star_u)
        D_U_adaptive = sp.diags(user_weights)
        item_weights = np.power(d_i + self.eps, -0.5)
        D_I_inv_half = sp.diags(item_weights)

        G_mid = X_sp.T @ D_U_adaptive @ X_sp
        G_tilde = (D_I_inv_half @ G_mid @ D_I_inv_half).toarray()

        # 5. Solve EASE closed-form
        G_tilde[np.diag_indices_from(G_tilde)] += self.reg_lambda
        P_np = np.linalg.inv(G_tilde)
        P_diag = np.diag(P_np)
        W_np = -P_np / (P_diag[np.newaxis, :] + self.eps)
        np.fill_diagonal(W_np, 0)

        # 6. Store on device
        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        self.train_matrix_gpu = self.get_train_matrix(data_loader)
        print("Adaptive ASPIRE fitting complete.")

    def forward(self, user_indices):
        input_tensor = torch.index_select(self.train_matrix_gpu, 0, user_indices).to_dense()
        return input_tensor @ self.weight_matrix
