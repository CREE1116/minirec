import torch
import torch.nn as nn
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
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting HybridWiener (alpha={self.alpha}, gamma={self.gamma}, lambda={self.reg_lambda}) on {self.device}...")
        self.train_matrix = self.get_train_matrix(data_loader)
        X = self.train_matrix
        
        # Step 1: Gram matrix G
        G = torch.sparse.mm(X.t(), X.to_dense())
        
        # Step 2: Statistics
        d = G.diagonal()                               # 1st order: Popularity
        e = torch.sqrt(torch.sum(torch.square(G), dim=1))    # 2nd order: Energy
        
        # Step 3: Hybrid Mixing
        # Use linear interpolation to balance spread and intensity
        s = (1.0 - self.gamma) * d + self.gamma * e
        
        # Step 4: Symmetric Normalization
        s_inv = 1.0 / (torch.pow(s + self.eps, self.alpha))
        G_tilde = G * s_inv.unsqueeze(1) * s_inv.unsqueeze(0)
        
        # Step 5: Solve (G_tilde + lambda * I) B = G_tilde
        A = G_tilde.clone()
        A.diagonal().add_(self.reg_lambda)
        
        self.weight_matrix = torch.linalg.solve(A, G_tilde)
        print("HybridWiener fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense()
        user_vec = self.train_matrix_dense[user_indices]
        return user_vec @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
