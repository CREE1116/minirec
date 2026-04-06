import torch
import numpy as np
from .base import BaseModel
from src.utils.svd import get_svd_cache

class PureSVD(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        # HPO에서 'k'를 제안할 경우 이를 최우선으로 사용 (embedding_dim보다 우선)
        self.embedding_dim = config['model'].get('k', config['model'].get('embedding_dim', 64))
        self.user_factors = None
        self.item_factors = None

    def fit(self, data_loader):
        # SVD 캐시 활용 (기본 1000차원까지 캐싱)
        k_cache = self.config.get('svd_cache_k', 1000)
        svd_data = get_svd_cache(data_loader, k_max=k_cache)
        
        # 현재 요청된 k만큼 슬라이싱 (Truncate)
        k = min(self.embedding_dim, len(svd_data['s']))
        print(f"Fitting PureSVD by truncating cached SVD (requested k={self.embedding_dim}, actual k={k})...")
        
        u = svd_data['u'][:, :k]
        s = svd_data['s'][:k]
        vt = svd_data['vt'][:k, :]
        
        # Store as torch tensors
        self.user_factors = torch.from_numpy(u * s).float().to(self.device) # U * S
        self.item_factors = torch.from_numpy(vt.T).float().to(self.device)  # V

    def forward(self, user_indices):
        u_f = self.user_factors[user_indices]
        return u_f @ self.item_factors.t()

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
