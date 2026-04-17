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
        print(f"Fitting Vectorized PPMI ASPIRE (alpha={self.alpha}) on CPU...")

        # 1. Load sparse matrix
        X = get_train_matrix_scipy(data_loader).tocsr()
        self.train_matrix_cpu = X # Store for hybrid inference
        n_users, n_items = X.shape

        # 2. co-occurrence matrix (sparse 유지)
        print("  Computing co-occurrence (CPU Sparse)...")
        G = (X.T @ X).tocoo()

        # 3. marginals (float32 강제)
        item_deg = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
        total = item_deg.sum().astype(np.float32)

        P_i = (item_deg / (total + self.eps)).astype(np.float32)

        # 4. vectorized PMI (NO LOOP, float32 강제)
        print("  Computing PMI (vectorized float32)...")

        i = G.row
        j = G.col
        v = G.data.astype(np.float32)

        # P(i,j)
        P_ij = (v / (total + self.eps)).astype(np.float32)

        # denominator: P(i) * P(j)^alpha
        denom = (P_i[i] * np.power(P_i[j], self.alpha)) + self.eps

        # log promotion 방지
        pmi = (np.log(P_ij + self.eps) - np.log(denom)).astype(np.float32)

        # 5. build sparse matrix directly
        W_sp = sp.coo_matrix((pmi, (i, j)), shape=(n_items, n_items), dtype=np.float32)

        # 6. symmetrization
        W_sp = (W_sp + W_sp.T) * np.float32(0.5)

        # 7. diagonal removal
        W_sp.setdiag(0)
        W_sp.eliminate_zeros()

        # 8. convert to torch (Hybrid Inference)
        # float32 보장하여 toarray()
        self.weight_matrix = torch.tensor(
            W_sp.toarray().astype(np.float32),
            dtype=torch.float32,
            device=self.device
        )
        
        del G, item_deg, P_i, i, j, v, P_ij, denom, pmi, W_sp
        gc.collect()

        print("Vectorized PPMI ASPIRE complete.")

    def forward(self, user_indices):
        return self._get_batch_ratings(user_indices, self.weight_matrix)
