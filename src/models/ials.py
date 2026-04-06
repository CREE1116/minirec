import torch
import torch.nn as nn
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
        
        # Initialize with small random values
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def fit(self, data_loader):
        print(f"Fitting iALS (dim={self.embedding_dim}, alpha={self.alpha}) on {self.device}...")
        
        # X is (n_users, n_items) sparse COO
        X = self.get_train_matrix(data_loader)
        
        # Confidence C = 1 + alpha * X
        # Preference P = X (since X is binary 1s)
        
        # Pre-calculate some values
        Y = self.item_embedding.weight # (n_items, dim)
        U = self.user_embedding.weight # (n_users, dim)
        
        for iteration in range(self.max_iter):
            # 1. Update Users
            # U = (Y^T C_u Y + lambda I)^-1 Y^T C_u P_u
            # Y^T C_u Y = Y^T Y + Y^T (C_u - I) Y = Y^T Y + alpha * Y^T diag(X_u) Y
            YTY = Y.t() @ Y
            
            # Since we iterate over users, we can use the "implicit" trick:
            # For each user u:
            # (YTY + alpha * Y_u^T Y_u + lambda I) u = (1 + alpha) * Y_u^T
            # where Y_u are the embeddings of items user u interacted with.
            
            # Batch update users
            # This can be slow if done one by one.
            # We can use the fact that X is sparse.
            
            # To speed up, we can use a block-diagonal approach or just iterate.
            # In PyTorch, we can use torch.linalg.solve for the whole batch if dim is small.
            # But the matrix is different for each user.
            
            # Optimized batch update:
            U = self._als_step(X, U, Y, is_user=True)
            self.user_embedding.weight.data.copy_(U)
            
            # 2. Update Items
            Y = self._als_step(X, Y, U, is_user=False)
            self.item_embedding.weight.data.copy_(Y)
            
            print(f"  Iteration {iteration+1}/{self.max_iter} complete.")

        print("iALS fitting complete.")

    def _als_step(self, X, factors, fixed_factors, is_user=True):
        """
        One step of ALS. 
        If is_user=True: updates U using fixed Y.
        If is_user=False: updates Y using fixed U.
        """
        n_factors = self.embedding_dim
        reg_id = torch.eye(n_factors, device=self.device) * self.reg_lambda
        
        # FTF = fixed_factors.T @ fixed_factors
        FTF = fixed_factors.t() @ fixed_factors
        
        # We need to solve: (FTF + alpha * F_i^T F_i + lambda I) x_i = (1 + alpha) * \sum F_i
        # where F_i are fixed factors for items/users i interacted with.
        
        new_factors = torch.zeros_like(factors)
        
        # To make it efficient in PyTorch without a custom CUDA kernel:
        # We can group users by their number of interactions, but that's complex.
        # Or just use a loop if n_users is not too large.
        # For better performance, we'll use the sparse matrix structure.
        
        # Extract indices and values from sparse X
        indices = X.indices()
        if not is_user:
            indices = indices.flip(0) # (item_id, user_id)
            
        # Sort indices to group by "owner" (user or item)
        # indices[0] is the owner_id
        owner_ids = indices[0]
        other_ids = indices[1]
        
        # Use a loop over owners. To speed up, we can process in large batches.
        # However, since each owner has a different number of "other" factors, 
        # it's hard to vectorize perfectly in vanilla PyTorch.
        
        # Let's use a reasonably efficient loop:
        unique_owners, counts = torch.unique_consecutive(owner_ids, return_counts=True)
        curr_idx = 0
        
        for owner_id, count in zip(unique_owners, counts):
            others = other_ids[curr_idx : curr_idx + count]
            F_i = fixed_factors[others] # (count, dim)
            
            # LHS = FTF + alpha * F_i.T @ F_i + lambda I
            LHS = FTF + self.alpha * (F_i.t() @ F_i) + reg_id
            
            # RHS = (1 + alpha) * \sum F_i
            RHS = (1 + self.alpha) * F_i.sum(dim=0)
            
            # Solve
            new_factors[owner_id] = torch.linalg.solve(LHS, RHS)
            curr_idx += count
            
        return new_factors

    def forward(self, user_indices):
        u_emb = self.user_embedding(user_indices)
        return u_emb @ self.item_embedding.weight.t()

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
