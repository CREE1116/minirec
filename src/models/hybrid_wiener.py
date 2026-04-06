import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel

class HybridWiener(BaseModel):
    """
    Hybrid 1st & 2nd Order Statistic Wiener Filter.
    Interpolates between Popularity (1st) and Energy (2nd) for better MNAR control.
    s = (1 - gamma) * diag(G) + gamma * RowEnergy(G)
    G_tilde = s^-alpha * G * s^-alpha
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.alpha = config['model'].get('alpha', 0.5) 
        self.gamma = config['model'].get('gamma', 0.5) # Mixing ratio (0: Pop, 1: Energy)
        self.eps = 1e-12
        
        self.weight_matrix = None
        self.train_matrix = sp.csr_matrix((self.n_users, self.n_items))

    def fit(self, data_loader):
        print(f"Fitting HybridWiener (alpha={self.alpha}, gamma={self.gamma}, lambda={self.reg_lambda})...")
        train_df = data_loader.train_df
        rows, cols = train_df['user_id'].values, train_df['item_id'].values
        X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), 
                          shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix = X
        
        # Step 1: Gram matrix G
        G = (X.T @ X).toarray()
        
        # Step 2: Statistics
        d = np.diag(G)                               # 1st order: Popularity
        e = np.sqrt(np.sum(np.square(G), axis=1))    # 2nd order: Energy
        
        # Step 3: Hybrid Mixing
        # Use linear interpolation to balance spread and intensity
        s = (1.0 - self.gamma) * d + self.gamma * e
        
        # Step 4: Symmetric Normalization
        s_inv = 1.0 / (np.power(s + self.eps, self.alpha))
        G_tilde = G * s_inv[:, np.newaxis] * s_inv[np.newaxis, :]
        
        # Step 5: Solve (G_tilde + lambda * I) B = G_tilde
        A = G_tilde.copy()
        np.fill_diagonal(A, np.diag(A) + self.reg_lambda)
        
        B = np.linalg.solve(A, G_tilde)
        
        self.weight_matrix = torch.from_numpy(B).float().to(self.device)
        print("HybridWiener fitting complete.")

    def forward(self, user_indices):
        u_ids = user_indices.cpu().numpy()
        user_vec = torch.from_numpy(self.train_matrix[u_ids].toarray()).float().to(self.device)
        return user_vec @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
