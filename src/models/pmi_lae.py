import numpy as np
import torch
import scipy.sparse as sp
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy

class PMILAE(BaseModel):
    """
    PMI-LAE: Linear AutoEncoder on top of a PPMI (Positive Pointwise Mutual Information) Kernel.
    Uses strict EASE closed-form solution to enforce diag(W)=0.
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.alpha = config['model'].get('alpha', 1.0)
        self.reg_lambda = config['model'].get('reg_lambda', 10.0)
        self.eps = 1e-12
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting PMI-LAE (alpha={self.alpha}, lambda={self.reg_lambda}) on CPU...")

        # 1. Load sparse matrix
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr() # Hybrid inference
        n_users, n_items = X_sp.shape

        # 2. Compute co-occurrence matrix G = X^T X
        print("  Computing co-occurrence (CPU Sparse)...")
        G = (X_sp.T @ X_sp).tocoo()

        # 3. Marginals
        item_deg = np.asarray(X_sp.sum(axis=0)).ravel()
        total = item_deg.sum()
        P_i = item_deg / (total + self.eps)

        # 4. Vectorized PMI calculation
        print("  Computing PPMI Kernel...")
        i, j, v = G.row, G.col, G.data
        
        P_ij = v / (total + self.eps)
        denom = (P_i[i] * np.power(P_i[j], self.alpha)) + self.eps
        
        pmi_values = np.log(P_ij + self.eps) - np.log(denom)
        pmi_values = np.maximum(pmi_values, 0)
        
        K_sp = sp.coo_matrix((pmi_values, (i, j)), shape=(n_items, n_items))
        K_sp = (K_sp + K_sp.T) * 0.5
        K_sp.setdiag(0)
        K_sp.eliminate_zeros()
        
        K_np = K_sp.toarray().astype(np.float32)
        del G, item_deg, P_i, i, j, v, P_ij, denom, pmi_values, K_sp
        gc.collect()
        
        # 5. Solve Strict EASE on top of PPMI Kernel (CPU NumPy)
        print("  Solving Strict EASE closed-form (CPU NumPy)...")
        P = K_np 
        P[np.diag_indices_from(P)] += self.reg_lambda
        
        try:
            P_inv = np.linalg.inv(P)
        except np.linalg.LinAlgError:
            P[np.diag_indices_from(P)] += 1e-4
            P_inv = np.linalg.inv(P)
        del P, K_np
        gc.collect()

        P_diag = np.diag(P_inv)
        W_np = -P_inv / (P_diag[np.newaxis, :] + self.eps)
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
