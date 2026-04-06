import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel

class EnergyWiener(BaseModel):
    """
    Degree-Ratio Symmetric Wiener Filter (GPU Accelerated).
    Combines User-Activity Ratio with Symmetric Normalization.
    Influence I = d^2 / S
    G_tilde = I^(-alpha/2) * G * I^(-alpha/2)
    W = (G_tilde + lambda * I)^-1 * G_tilde
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.alpha = config['model'].get('alpha', 0.5) 
        self.eps = 1e-12
        
        self.weight_matrix = None
        self.train_matrix = sp.csr_matrix((self.n_users, self.n_items))

    def fit(self, data_loader):
        print(f"Fitting EnergyWiener (GPU, alpha={self.alpha}, lambda={self.reg_lambda})...")
        train_df = data_loader.train_df
        rows, cols = train_df['user_id'].values, train_df['item_id'].values
        X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), 
                          shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix = X
        
        # Step 1: Gram matrix calculation (CPU Sparse -> Dense GPU)
        G_cpu = (X.T @ X).toarray()
        G = torch.from_numpy(G_cpu).to(self.device)
        
        # Step 2: Item Statistics on GPU
        d = torch.diag(G)             # Degree (Popularity)
        S = G.sum(dim=1)              # Row Sum (Contextual Volume)
        
        # Step 3: Compute Influence-based Normalization Factor
        # Influence I = d^2 / S (Degree-Ratio Logic)
        influence = (d**2) / (S + self.eps)
        
        # Symmetric Norm scaling: influence^(-alpha / 2)
        d_inv = torch.pow(influence + self.eps, -self.alpha / 2.0)
        
        # Step 4: Symmetric Normalization G_tilde = D_inv @ G @ D_inv
        G_tilde = G * d_inv.unsqueeze(1) * d_inv.unsqueeze(0)
        
        # Step 5: Wiener Filter Solve on GPU
        # (G_tilde + lambda * I) W = G_tilde
        A = G_tilde + self.reg_lambda * torch.eye(self.n_items, device=self.device)
        
        # torch.linalg.solve is significantly faster on GPU than NumPy on CPU
        self.weight_matrix = torch.linalg.solve(A, G_tilde)
        
        print("EnergyWiener GPU fitting complete.")

    def forward(self, user_indices):
        u_ids = user_indices.cpu().numpy()
        user_vec = torch.from_numpy(self.train_matrix[u_ids].toarray()).float().to(self.device)
        return user_vec @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
