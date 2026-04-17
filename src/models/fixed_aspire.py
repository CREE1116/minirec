import torch
import numpy as np
import scipy.sparse as sp
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class FixedAspire(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = np.float32(config['model'].get('reg_lambda', 10.0))
        self.alpha      = np.float32(config['model'].get('alpha', 1.0))
        self.eps        = np.float32(1e-12)
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting ASPIRE (alpha={self.alpha}) on CPU (Simple Style)...")

        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr()

        # Step 1: Pre-calculate Weights
        n_u = np.asarray(X_sp.sum(axis=1)).ravel().astype(np.float32)
        u_weights = (np.float32(1.0) / (np.power(n_u, self.alpha) + self.eps)).astype(np.float32)

        n_i = np.asarray(X_sp.sum(axis=0)).ravel().astype(np.float32)
        i_weights = np.power(n_i + self.eps, -self.alpha/np.float32(2.0)).astype(np.float32)
        
        del n_u, n_i
        gc.collect()

        # Step 2: Gram Matrix (Direct Simple)
        print("  computing gram matrix...")
        G_np = compute_gram_matrix(X_sp, data_loader, weights=u_weights, item_weights=i_weights)
        
        del u_weights, i_weights
        gc.collect()

        # Step 3: Inversion (NumPy inv)
        print("  inverting matrix (NumPy)...")
        G_np[np.diag_indices_from(G_np)] += self.reg_lambda
        P_np = np.linalg.inv(G_np).astype(np.float32)
        del G_np
        gc.collect()

        # Step 4: Weights
        diag_P = np.diag(P_np).astype(np.float32)
        W_np = (-P_np / (diag_P[np.newaxis, :] + self.eps)).astype(np.float32)
        np.fill_diagonal(W_np, 0)
        del P_np, diag_P

        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        del W_np
        gc.collect()

        if 'cuda' in str(self.device): torch.cuda.empty_cache()
        print("ASPIRE fitting complete.")

    def forward(self, user_indices):
        return self._get_batch_ratings(user_indices, self.weight_matrix)
