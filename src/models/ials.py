import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from implicit.cpu.als import AlternatingLeastSquares
from .base import BaseModel

class iALS(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.embedding_dim = config['model'].get('embedding_dim', 128)
        self.reg_lambda = config['model'].get('reg_lambda', 0.01)
        self.alpha = config['model'].get('alpha', 40.0)
        self.max_iter = config['model'].get('max_iter', 15)
        
        # Factors stored in torch embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)

    def fit(self, data_loader):
        print(f"Fitting iALS (dim={self.embedding_dim}, alpha={self.alpha})...")
        train_df = data_loader.train_df
        # Implicit expects (item x user) or (user x item) depending on the setup.
        # CSR matrix: (user, item)
        X = sp.csr_matrix((np.ones(len(train_df), dtype=np.float32), (train_df['user_id'], train_df['item_id'])), 
                          shape=(self.n_users, self.n_items))
        
        # ALS Model
        model = AlternatingLeastSquares(
            factors=self.embedding_dim,
            regularization=self.reg_lambda,
            alpha=self.alpha,
            iterations=self.max_iter,
            random_state=self.config.get('seed', 42),
            # use_gpu=False # CPU for stability in minirec
        )
        model.fit(X, show_progress=True)
        
        # Copy to torch
        with torch.no_grad():
            self.user_embedding.weight.copy_(torch.from_numpy(model.user_factors))
            self.item_embedding.weight.copy_(torch.from_numpy(model.item_factors))
        
        print("iALS fitting complete.")

    def forward(self, user_indices):
        u_emb = self.user_embedding(user_indices)
        return u_emb @ self.item_embedding.weight.t()

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
