import torch
import torch.nn as nn
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
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting EnergyWiener (alpha={self.alpha}, lambda={self.reg_lambda}) on {self.device}...")
        self.train_matrix = self.get_train_matrix(data_loader)
        X = self.train_matrix
        
        # Step 1: Gram matrix calculation (Sparse -> Dense GPU)
        G = torch.sparse.mm(X.t(), X.to_dense())
        
        # Step 2: Item Statistics on GPU
        d = G.diagonal()             # Degree (Popularity)
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
        A = G_tilde.clone()
        A.diagonal().add_(self.reg_lambda)
        
        # torch.linalg.solve is significantly faster on GPU than NumPy on CPU
        self.weight_matrix = torch.linalg.solve(A, G_tilde)
        
        print("EnergyWiener fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense()
        user_vec = self.train_matrix_dense[user_indices]
        return user_vec @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
