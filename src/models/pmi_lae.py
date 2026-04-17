import numpy as np
import torch
import scipy.sparse as sp
import scipy.linalg as la
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class PMILAE(BaseModel):
    """
    PMI-LAE: Linear AutoEncoder on top of a PPMI (Positive Pointwise Mutual Information) Kernel.
    Uses strict EASE closed-form solution with In-place float32 optimization.
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.alpha = np.float32(config['model'].get('alpha', 1.0))
        self.reg_lambda = np.float32(config['model'].get('reg_lambda', 10.0))
        self.eps = np.float32(1e-12)
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting PMI-LAE (alpha={self.alpha}, lambda={self.reg_lambda}) on CPU with Strict float32...")

        # 1. Load sparse matrix
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr() # Hybrid inference
        n_users, n_items = X_sp.shape

        # 2. Compute co-occurrence matrix G = X^T X
        print("  Computing co-occurrence (CPU Sparse)...")
        G = (X_sp.T @ X_sp).tocoo()

        # 3. Marginals
        item_deg = np.asarray(X_sp.sum(axis=0)).ravel().astype(np.float32)
        total = item_deg.sum().astype(np.float32)
        P_i = (item_deg / (total + self.eps)).astype(np.float32)

        # 4. Vectorized PMI calculation (Strict float32)
        print("  Computing PPMI Kernel (float32)...")
        i, j, v = G.row, G.col, G.data.astype(np.float32)
        
        P_ij = (v / (total + self.eps)).astype(np.float32)
        denom = ((P_i[i] * np.power(P_i[j], self.alpha)) + self.eps).astype(np.float32)
        
        # Prevent promotion during log
        pmi_values = (np.log(P_ij + self.eps) - np.log(denom)).astype(np.float32)
        pmi_values = np.maximum(pmi_values, 0).astype(np.float32)
        
        K_sp = sp.coo_matrix((pmi_values, (i, j)), shape=(n_items, n_items), dtype=np.float32)
        K_sp = (K_sp + K_sp.T) * np.float32(0.5)
        K_sp.setdiag(0)
        K_sp.eliminate_zeros()
        
        K_np = K_sp.toarray().astype(np.float32)
        del G, item_deg, P_i, i, j, v, P_ij, denom, pmi_values, K_sp
        gc.collect()
        
        # 5. Solve Strict EASE on top of PPMI Kernel (CPU In-place float32)
        print("  Solving Strict EASE closed-form (CPU In-place NumPy float32)...")
        # G_np is already a fresh copy
        K_np[np.diag_indices_from(K_np)] += self.reg_lambda
        
        P_inv = la.inv(K_np, overwrite_a=True).astype(np.float32)
        del K_np
        gc.collect()

        P_diag = np.diag(P_inv).astype(np.float32)
        W_np = (-P_inv / (P_diag[np.newaxis, :] + self.eps)).astype(np.float32)
        np.fill_diagonal(W_np, 0)
        
        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        del P_inv, W_np
        
        gc.collect()
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        print("PMI-LAE fitting complete.")

    def forward(self, user_indices):
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
