import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel

class DLAE(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.dropout_p = config['model'].get('dropout_p', 0.5)
        self.weight_matrix = None
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting DLAE (p={self.dropout_p}, lambda={self.reg_lambda})...")
        train_df = data_loader.train_df
        rows, cols = train_df['user_id'].values, train_df['item_id'].values
        X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), 
                          shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix = X
        
        # Step 1: G = X^T X
        # scipy.sparse X.T @ X is efficient
        G = (X.T @ X).toarray()
        
        # Step 2: g_diag = diag(G)
        g_diag = np.diag(G)
        
        # Step 3: w = (p / (1-p)) * g_diag
        p = self.dropout_p
        if p >= 1.0: p = 0.99 # Safety clamp
        w = (p / (1.0 - p)) * g_diag
        
        # Step 4: Lambda = diag(w) + lambda * I
        L = np.diag(w + self.reg_lambda)
        
        # Step 5: Solve (G + Lambda) B = G
        # G + Lambda = G + diag(w + reg_lambda)
        # Note: Step 4 and 5 can be merged as G += diag(w + reg_lambda)
        # then solve (G) B = (G_original)
        
        G_lhs = G + L
        # Solve G_lhs * B = G
        B = np.linalg.solve(G_lhs, G)
        
        self.weight_matrix = torch.from_numpy(B).float().to(self.device)
        print("DLAE fitting complete.")

    def forward(self, user_indices):
        u_ids = user_indices.cpu().numpy()
        user_vec = torch.from_numpy(self.train_matrix[u_ids].toarray()).float().to(self.device)
        return user_vec @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
