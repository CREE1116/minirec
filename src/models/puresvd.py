import torch
import numpy as np
from .base import BaseModel
from src.utils.svd import get_svd_cache

class PureSVD(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.embedding_dim = config['model'].get('k', config['model'].get('embedding_dim', 64))
        self.user_factors = None
        self.item_factors = None

    def fit(self, data_loader):
        k_cache = self.config.get('svd_cache_k', 1000)
        svd_data = get_svd_cache(data_loader, k_max=k_cache)

        k = min(self.embedding_dim, len(svd_data['s']))
        print(f"Fitting PureSVD (k={k}) on {self.device}...")

        u = svd_data['u'][:, :k]
        s = svd_data['s'][:k]
        vt = svd_data['vt'][:k, :]

        self.user_factors = torch.from_numpy(u * s).float().to(self.device)
        self.item_factors = torch.from_numpy(vt.T).float().to(self.device)

    def forward(self, user_indices):
        return self.user_factors[user_indices] @ self.item_factors.t()

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
