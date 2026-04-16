import numpy as np
import torch
import scipy.sparse as sp
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy


class PMIAspire(BaseModel):

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)

        self.alpha = config['model'].get('alpha', 0.75)
        self.eps = 1e-12
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting Vectorized PPMI ASPIRE (alpha={self.alpha})...")

        # 1. Load sparse matrix
        X = get_train_matrix_scipy(data_loader).tocsr()
        n_users, n_items = X.shape

        # 2. co-occurrence matrix (sparse 유지)
        print("  Computing co-occurrence...")
        G = (X.T @ X).tocoo()

        # 3. marginals
        item_deg = np.asarray(X.sum(axis=0)).ravel()
        total = item_deg.sum()

        P_i = item_deg / (total + self.eps)

        # 4. vectorized PMI (NO LOOP)
        print("  Computing PMI (vectorized)...")

        i = G.row
        j = G.col
        v = G.data

        # P(i,j)
        P_ij = v / (total + self.eps)

        # denominator: P(i) * P(j)^alpha
        denom = (P_i[i] * np.power(P_i[j], self.alpha)) + self.eps

        pmi = np.log(P_ij + self.eps) - np.log(denom)

        # 5. build sparse matrix directly
        W = sp.coo_matrix((pmi, (i, j)), shape=(n_items, n_items))

        # 6. symmetrization (important but still vectorized)
        W = (W + W.T) * 0.5

        # 7. diagonal removal
        W.setdiag(0)
        W.eliminate_zeros()

        # 8. convert to torch
        self.weight_matrix = torch.tensor(
            W.toarray(),
            dtype=torch.float32,
            device=self.device
        )

        self.train_matrix_gpu = self.get_train_matrix(data_loader)

        print("Vectorized PPMI ASPIRE complete.")

    def forward(self, user_indices):
        X_u = torch.index_select(
            self.train_matrix_gpu, 0, user_indices
        ).to_dense()

        return X_u @ self.weight_matrix