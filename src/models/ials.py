import torch
import torch.nn as nn
from tqdm import tqdm
from .base import BaseModel

class iALS(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.embedding_dim = config['model'].get('embedding_dim', 128)
        self.reg_lambda = config['model'].get('reg_lambda', 0.01)
        self.alpha = config['model'].get('alpha', 40.0)
        self.max_iter = config['model'].get('max_iter', 15)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def fit(self, data_loader):
        print(f"Fitting iALS (dim={self.embedding_dim}, alpha={self.alpha}) on {self.device}...")
        X = self.get_train_matrix(data_loader)
        Y = self.item_embedding.weight
        U = self.user_embedding.weight

        for iteration in range(self.max_iter):
            U = self._als_step(X, U, Y, is_user=True,  desc=f"[{iteration+1}/{self.max_iter}] users")
            self.user_embedding.weight.data.copy_(U)
            Y = self._als_step(X, Y, U, is_user=False, desc=f"[{iteration+1}/{self.max_iter}] items")
            self.item_embedding.weight.data.copy_(Y)
        print("iALS fitting complete.")

    def _als_step(self, X, factors, fixed_factors, is_user=True, desc=""):
        reg_id = torch.eye(self.embedding_dim, device=self.device) * self.reg_lambda
        FTF = fixed_factors.t() @ fixed_factors
        new_factors = torch.zeros_like(factors)

        indices = X.indices()
        if not is_user:
            indices = indices.flip(0)

        owner_ids = indices[0]
        other_ids = indices[1].to(self.device)

        unique_owners, counts = torch.unique_consecutive(owner_ids, return_counts=True)
        curr_idx = 0
        for owner_id, count in tqdm(zip(unique_owners, counts), total=len(unique_owners), desc=desc, leave=False):
            others = other_ids[curr_idx : curr_idx + count]
            F_i = fixed_factors[others]
            LHS = FTF + self.alpha * (F_i.t() @ F_i) + reg_id
            RHS = (1 + self.alpha) * F_i.sum(dim=0)
            new_factors[owner_id] = torch.linalg.solve(LHS, RHS)
            curr_idx += count

        return new_factors

    def forward(self, user_indices):
        return self.user_embedding(user_indices) @ self.item_embedding.weight.t()

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
