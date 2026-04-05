import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from .base import BaseModel

class PureSVD(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        # HPO에서 'k'를 제안할 경우 이를 최우선으로 사용 (embedding_dim보다 우선)
        self.embedding_dim = config['model'].get('k', config['model'].get('embedding_dim', 64))
        self.user_factors = None
        self.item_factors = None

    def fit(self, data_loader):
        print(f"Fitting PureSVD (dim={self.embedding_dim})...")
        train_df = data_loader.train_df
        X = sp.csr_matrix((np.ones(len(train_df)), (train_df['user_id'], train_df['item_id'])), 
                          shape=(self.n_users, self.n_items), dtype=np.float32)
        
        # SVD: X \approx U * S * V^T
        # k는 min(dim, n_users-1, n_items-1)이어야 함
        k = min(self.embedding_dim, min(X.shape) - 1)
        u, s, vt = svds(X, k=k)
        
        # Sort by singular values descending
        idx = np.argsort(s)[::-1]
        u, s, vt = u[:, idx], s[idx], vt[idx, :]
        
        # Store as torch tensors
        self.user_factors = torch.from_numpy(u * s).float().to(self.device) # U * S
        self.item_factors = torch.from_numpy(vt.T).float().to(self.device)  # V

    def forward(self, user_indices):
        u_f = self.user_factors[user_indices]
        return u_f @ self.item_factors.t()

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
