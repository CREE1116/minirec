import numpy as np
import torch
import scipy.sparse as sp
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy

class PMIAspire(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.alpha = np.float32(config['model'].get('alpha', 0.75))
        self.eps = np.float32(1e-12)
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting PPMI ASPIRE (alpha={self.alpha}) on CPU with Block-wise float32...")

        # 1. Load sparse matrix
        X_sp = get_train_matrix_scipy(data_loader).tocsr()
        self.train_matrix_cpu = X_sp
        n_users, n_items = X_sp.shape

        # 2. Compute Marginals (float32)
        item_deg = np.asarray(X_sp.sum(axis=0)).ravel().astype(np.float32)
        total = item_deg.sum().astype(np.float32)
        P_i = (item_deg / (total + self.eps)).astype(np.float32)

        # 3. Block-wise PPMI Calculation (To avoid memory explosion)
        # We don't use compute_gram_matrix here because PPMI logic is different, 
        # but we use the same block-wise philosophy.
        W_np = np.zeros((n_items, n_items), dtype=np.float32)
        X_csc = X_sp.tocsc()
        X_T_csr = X_sp.T.tocsr()
        block_size = 4000

        print("  Computing PPMI Kernel via Blocks...")
        for start in range(0, n_items, block_size):
            end = min(start + block_size, n_items)
            
            # (Users, Block)
            X_block = X_csc[:, start:end].toarray().astype(np.float32)
            # G_block = X.T @ X_block (Co-occurrence)
            G_block = (X_T_csr @ X_block).astype(np.float32)
            
            # Convert to probabilities
            P_ij = G_block / (total + self.eps)
            # denom: P(i) * P(j)^alpha
            denom = (P_i[:, np.newaxis] * np.power(P_i[start:end], self.alpha)) + self.eps
            
            # PPMI = max(0, log(Pij/denom))
            pmi = (np.log(P_ij + self.eps) - np.log(denom)).astype(np.float32)
            pmi = np.maximum(pmi, 0)
            
            W_np[:, start:end] = pmi
            
            del X_block, G_block, P_ij, denom, pmi
            if start % (block_size * 4) == 0: gc.collect()

        # 4. Symmetrization & Diag Removal
        print("  Symmetrizing...")
        W_np = (W_np + W_np.T) * np.float32(0.5)
        np.fill_diagonal(W_np, 0)

        # 5. Result to GPU
        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        
        del W_np, item_deg, P_i
        gc.collect()

        print("Vectorized PPMI ASPIRE complete.")

    def forward(self, user_indices):
        return self._get_batch_ratings(user_indices, self.weight_matrix)
