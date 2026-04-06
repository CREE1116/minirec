import torch
import torch.nn as nn
from .base import BaseModel

class EASE(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.weight_matrix = None
        self.train_matrix = None # PyTorch Sparse (CSR-like) matrix for indexing

    def fit(self, data_loader):
        print(f"Fitting EASE (lambda={self.reg_lambda}) on {self.device}...")
        
        # Sparse interaction matrix X
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X # COO
        
        # G = X'X + lambda*I
        # Sparse @ Sparse in PyTorch is limited. 
        # But for EASE, G is n_items x n_items dense.
        # X is (users, items). X.T is (items, users).
        # We can use torch.sparse.mm for X.T @ X but result is dense
        
        # (items, users) @ (users, items) -> (items, items)
        # Using sparse.mm is memory efficient if X is sparse
        G = torch.sparse.mm(X.t(), X).to_dense()
        
        G.diagonal().add_(self.reg_lambda)
        
        # P = inv(G)
        P = torch.linalg.inv(G)
        
        # B = I - P / diag(P)
        # diag_P = P.diagonal()
        B = P / (-P.diagonal())
        B.diagonal().zero_()
        
        self.weight_matrix = B
        print("EASE fitting complete.")

    def forward(self, user_indices):
        # user_indices: torch.LongTensor on self.device
        # We need to slice self.train_matrix (sparse)
        # Slicing sparse COO in torch is not efficient, but we can do it via index_select if converted to CSR
        # Or we can just use dense matrix if items is small.
        # For now, let's use torch.index_select on dense version if memory allows, 
        # OR use a more efficient way for sparse indexing.
        
        # X is (n_users, n_items). We want X[user_indices]
        # Torch sparse COO doesn't support easy indexing like NumPy.
        # A common trick is to use embedding with sparse weight, but here we want rows of X.
        
        # Let's convert train_matrix to CSR for faster indexing if possible, 
        # but PyTorch CSR indexing is also limited.
        
        # Alternative: convert only the required rows to dense
        # For large scale, we should be careful.
        
        u_ids = user_indices
        # Create a dense representation of the batch
        # This is essentially what the previous numpy version did: self.train_matrix[u_ids].toarray()
        
        # Since self.train_matrix is sparse COO, we can't easily slice it.
        # Let's use a workaround:
        # 1. Convert to dense if small enough
        # 2. Or use torch.index_select on a dense matrix
        
        # If we have many items, dense might OOM. 
        # But if we are in this loop, we probably fit EASE which already required n_items^2 dense matrix.
        # So n_users x n_items might be large, but batch_size x n_items is usually fine.
        
        # Efficient way to get sparse rows as dense:
        # We can use the fact that train_matrix.indices() has user_id and item_id.
        
        # For simplicity and to maintain parity with numpy version:
        # user_vec = self.train_matrix.to_dense()[u_ids] # Slow if n_users is large
        
        # Better: extract only relevant indices
        indices = self.train_matrix.indices()
        values = self.train_matrix.values()
        
        # This is still tricky. Let's just use the fact that EASE fitting usually implies n_items is manageable.
        # If n_users is also manageable, we can store dense. 
        # If not, we can re-create sparse for each batch or use CSR.
        
        # Let's use a dense matrix for train_matrix if it fits in memory, 
        # or use a more optimized sparse row selection.
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense()
            
        user_vec = self.train_matrix_dense[u_ids]
        return user_vec @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
