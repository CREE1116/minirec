import torch
import numpy as np
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class EASE(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = np.float32(config['model'].get('reg_lambda', 500.0))
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting EASE (lambda={self.reg_lambda}) on CPU (Minimal Copy)...")
        
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr()

        # Step 1: Gram Matrix (Direct 8.3GB allocation)
        G_np = compute_gram_matrix(X_sp, data_loader)
        gc.collect()
        
        # Step 2: In-place diag add
        G_np[np.diag_indices_from(G_np)] += self.reg_lambda
        
        # Step 3: Inversion (np.linalg.inv always makes one copy)
        print("  inverting matrix (NumPy)...")
        P_np = np.linalg.inv(G_np)
        del G_np # Delete input immediately
        gc.collect()
        
        # Step 4: Final weights (In-place to save memory)
        diag_P = np.diag(P_np)
        # Use /= to avoid creating another 8.3GB matrix
        P_np /= -(diag_P + np.float32(1e-12))
        np.fill_diagonal(P_np, 0)
        
        self.weight_matrix = torch.tensor(P_np, dtype=torch.float32, device=self.device)
        del P_np
        
        gc.collect()
        if 'cuda' in str(self.device): torch.cuda.empty_cache()
        print("EASE fitting complete.")

    def forward(self, user_indices):
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
