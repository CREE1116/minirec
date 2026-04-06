import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from .base import BaseModel

class iALS(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.embedding_dim = config['model'].get('embedding_dim', 128)
        self.reg_lambda    = config['model'].get('reg_lambda', 0.01)
        self.alpha         = config['model'].get('alpha', 40.0)
        self.max_iter      = config['model'].get('max_iter', 15)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)

    def fit(self, data_loader):
        use_gpu = self.device.type == 'cuda'
        print(f"Fitting iALS (dim={self.embedding_dim}, alpha={self.alpha}) via implicit (gpu={use_gpu})...")

        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        vals = np.ones(len(rows), dtype=np.float32)

        # implicit 0.7+: fit()은 user×item CSR 행렬을 받음
        user_items = sp.csr_matrix((vals, (rows, cols)), shape=(self.n_users, self.n_items))

        def _make_model(use_gpu):
            return AlternatingLeastSquares(
                factors=self.embedding_dim,
                regularization=self.reg_lambda,
                alpha=self.alpha,
                iterations=self.max_iter,
                use_gpu=use_gpu,
            )

        try:
            model = _make_model(use_gpu)
        except ValueError as e:
            if use_gpu and 'CUDA' in str(e):
                print(f"[iALS] GPU extension unavailable, falling back to CPU...")
                model = _make_model(False)
            else:
                raise
        model.fit(user_items)

        self.user_embedding.weight.data.copy_(
            torch.from_numpy(model.user_factors).to(self.device))
        self.item_embedding.weight.data.copy_(
            torch.from_numpy(model.item_factors).to(self.device))
        print("iALS fitting complete.")

    def forward(self, user_indices):
        return self.user_embedding(user_indices) @ self.item_embedding.weight.t()

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
