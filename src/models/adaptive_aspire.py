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
        self.train_matrix_cpu = None # Keep large matrix on CPU

    def _to_torch_sparse(self, scipy_matrix):
        """Convert Scipy sparse matrix to Torch sparse tensor (avoiding toarray)"""
        scipy_matrix = scipy_matrix.tocoo()
        indices = torch.from_numpy(np.vstack((scipy_matrix.row, scipy_matrix.col)).astype(np.int64))
        values = torch.from_numpy(scipy_matrix.data.astype(np.float32))
        return torch.sparse_coo_tensor(indices, values, torch.Size(scipy_matrix.shape))

    def fit(self, data_loader):
        print("Fitting Adaptive ASPIRE (Optimized Memory)...")

        # 1. Get training data as Scipy CSR (Keep on CPU)
        X_sp = get_train_matrix_scipy(data_loader) 
        self.train_matrix_cpu = X_sp.tocsr() # Store for inference
        U, I = X_sp.shape

        # 2. Compute Proxies
        d_u = np.asarray(X_sp.sum(axis=1)).ravel()
        d_i = np.asarray(X_sp.sum(axis=0)).ravel()
        
        # 3. Calculate Adaptive Gamma*
        p_u_proxy = (d_u + self.eps) / (d_u.max() + self.eps)
        d_bar_u = np.asarray(X_sp.dot(d_i)).ravel() / (d_u + self.eps)
        
        valid = (d_u > 1) & (d_bar_u > 1)
        gamma_star_u = np.where(valid, -np.log(p_u_proxy) / np.log(d_bar_u + self.eps), 0.0)
        
        # 4. Statistical Noise Estimation (O(N) CPU)
        avg_d = np.mean(d_u)
        var_d = np.var(d_u)
        density = avg_d / I
        expected_var_random = avg_d * (1 - density)
        noise_ratio = expected_var_random / (var_d + expected_var_random + self.eps)
        sigma_est = noise_ratio * 0.5 
        
        damping_factor = max(0.1, 1.0 - self.damping_coeff * sigma_est)
        gamma_star_u = (gamma_star_u * damping_factor).clip(0.0, 1.0)
        
        print(f"  -> Damping Factor: {damping_factor:.4f}, Mean Gamma: {np.mean(gamma_star_u):.4f}")
        
        # 5. Construct Adaptive Gram Matrix (CPU Sparse Multiplication)
        user_weights = np.power(d_u + self.eps, -gamma_star_u)
        D_U_adaptive = sp.diags(user_weights)
        item_weights = np.power(d_i + self.eps, -0.5)
        D_I_inv_half = sp.diags(item_weights)

        # X.T @ D_U @ X is computed efficiently on CPU
        G_mid = X_sp.T @ D_U_adaptive @ X_sp
        G_tilde_cpu = (D_I_inv_half @ G_mid @ D_I_inv_half).toarray()

        # 6. Solve Ridge on GPU (Fast Inversion)
        print(f"  Solving Ridge Regression on {self.device}...")
        G_torch = torch.from_numpy(G_tilde_cpu).to(torch.float32).to(self.device)
        G_torch.diagonal().add_(self.reg_lambda)
        
        P = torch.linalg.inv(G_torch)
        
        # EASE-style weight derivation
        P_diag = torch.diagonal(P)
        W_gpu = -P / (P_diag.unsqueeze(0) + self.eps)
        W_gpu.diagonal().zero_()

        # 7. Store weights on GPU
        self.weight_matrix = W_gpu
        print("Adaptive ASPIRE fitting complete.")

    def forward(self, user_indices):
        """
        Memory Efficient Inference: Slice sparse on CPU, Multiply on GPU
        """
        user_ids_np = user_indices.cpu().numpy()
        # 1. Slice on CPU (Fast)
        batch_sp = self.train_matrix_cpu[user_ids_np]
        # 2. Move to GPU as Sparse
        batch_torch = self._to_torch_sparse(batch_sp).to(self.device)
        # 3. Sparse-Dense Multiplication (Efficient)
        return torch.sparse.mm(batch_torch, self.weight_matrix)
