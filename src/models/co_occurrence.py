import torch
import numpy as np
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy

class CoOccurrence(BaseModel):
    """
    Simple Co-occurrence based Recommendation.
    Weight Matrix W = X^T @ X, with diag(W) = 0.
    Score = X @ W
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.weight_matrix = None

    def fit(self, data_loader):
        print("Fitting Simple Co-occurrence model...")
        
        # 1. Load interaction matrix X (U x I)
        X_sp = get_train_matrix_scipy(data_loader)
        
        # 2. Compute Item-Item Co-occurrence matrix: W = X^T @ X
        print("  Computing X^T @ X...")
        W_sp = X_sp.T @ X_sp
        
        # 3. Convert to dense and zero out diagonal
        W_np = W_sp.toarray().astype(np.float32)
        np.fill_diagonal(W_np, 0)
        
        # 4. Normalize rows to [0, 1] relative to max co-occurrence to avoid extreme scores
        # (This makes it more like a similarity matrix)
        row_max = W_np.max(axis=1, keepdims=True)
        W_np = W_np / (row_max + 1e-12)

        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        self.train_matrix_gpu = self.get_train_matrix(data_loader)
        print("Co-occurrence fitting complete.")

    def forward(self, user_indices):
        # r_u @ W
        input_tensor = torch.index_select(self.train_matrix_gpu, 0, user_indices).to_dense()
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
