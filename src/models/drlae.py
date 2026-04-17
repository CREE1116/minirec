import torch
import numpy as np
import scipy.linalg as la
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class DRLAE(BaseModel):
    """
    Doubly Robust Linear AutoEncoder (DRLAE)
    Optimized with Strict Minimal Memory and Block-wise construction.
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        
        model_cfg = config.get('model', {})
        self.lambda_base = np.float32(model_cfg.get('reg_lambda', 500.0))
        self.lambda_var = np.float32(model_cfg.get('lambda_var', 1.0))
        self.min_prop = np.float32(model_cfg.get('min_prop', 1e-4))
        self.gamma_val = np.float32(model_cfg.get('gamma', 0.5)) 
        self.eps = np.float32(1e-12)
        
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting DRLAE on CPU with Strict minimal memory...")
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr() # Hybrid inference
        
        # 1. Propensity Score (p_i)
        item_counts = np.array(X_sp.sum(axis=0)).flatten().astype(np.float32)
        p_vec = (item_counts / (item_counts.max() + self.eps)) ** self.gamma_val
        p_vec = np.clip(p_vec, a_min=self.min_prop, a_max=None).astype(np.float32)
        
        # 2. Optimized Gram matrix calculation (Block-wise)
        # S_star = S / (p_i * p_j)
        # item_weights for compute_gram_matrix applies D_i @ G @ D_i
        # So we pass 1/p_vec as item_weights
        print("  Computing Weighted Gram matrix (Block-wise CPU)...")
        inv_p_vec = (np.float32(1.0) / p_vec).astype(np.float32)
        G_np = compute_gram_matrix(X_sp, data_loader, item_weights=inv_p_vec)
        
        # 4. Variance-Penalized Regularization
        print("  Applying variance penalty...")
        # S_diag is diag of original G
        S_diag = item_counts.astype(np.float32) 
        var_penalty = ((np.float32(1.0) - p_vec**2) / (p_vec**4 + self.eps)) * (S_diag**2)
        omega_diag = self.lambda_base + self.lambda_var * var_penalty
        
        G_np[np.diag_indices_from(G_np)] += omega_diag
        
        # 5. In-place Inversion
        print("  Solving linear system (CPU In-place float32)...")
        try:
            P_inv = la.inv(G_np, overwrite_a=True).astype(np.float32)
        except (np.linalg.LinAlgError, la.LinAlgError):
            print("[Warning] Singular matrix, using stronger regularization.")
            G_np[np.diag_indices_from(G_np)] += self.lambda_base * np.float32(10)
            P_inv = la.inv(G_np, overwrite_a=True).astype(np.float32)
        
        del G_np, item_counts, p_vec, S_diag, var_penalty, omega_diag, inv_p_vec
        gc.collect()

        # 6. Final W 도출
        diag_P = np.diag(P_inv).astype(np.float32)
        W_np = (-P_inv / (diag_P[np.newaxis, :] + self.eps)).astype(np.float32)
        np.fill_diagonal(W_np, 0)
        
        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        del P_inv, W_np, diag_P
        
        gc.collect()
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        print("DRLAE fitting complete.")

    def forward(self, user_indices):
        return self._get_batch_ratings(user_indices, self.weight_matrix)
    
    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
