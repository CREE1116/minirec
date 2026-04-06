import torch
import torch.nn as nn
from .base import BaseModel

class EnergyDLAE(BaseModel):
    """
    Energy-based Dropout Latent AutoEncoder.
    Combines DLAE's dropout-based closed-form solution with 2nd-order energy statistics.
    1. G_tilde = E^-alpha * G * E^-alpha  (Symmetric Energy Normalization)
    2. Regularization = (p/(1-p)) * diag(Energy_tilde) + lambda * I
    3. W = (G_tilde + Regularization)^-1 * G_tilde
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.dropout_p = config['model'].get('dropout_p', 0.5)
        self.alpha = config['model'].get('alpha', 0.5) # Energy normalization strength
        self.eps = 1e-12
        
        self.weight_matrix = None
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting EnergyDLAE (p={self.dropout_p}, alpha={self.alpha}, lambda={self.reg_lambda}) on {self.device}...")
        self.train_matrix = self.get_train_matrix(data_loader)
        X = self.train_matrix
        
        # Step 1: Gram matrix G = X.T @ X
        G = torch.sparse.mm(X.t(), X.to_dense())
        
        # Step 2: 2nd-Order Statistic - Row Energy of G
        row_energy = torch.sqrt(torch.sum(torch.square(G), dim=1))
        
        # Step 3: Symmetric Energy Normalization
        e_inv = 1.0 / (torch.pow(row_energy + self.eps, self.alpha))
        G_tilde = G * e_inv.unsqueeze(1) * e_inv.unsqueeze(0)
        
        # Step 4: DLAE-style Diagonal Regularization using Energy of G_tilde
        g_tilde_diag = G_tilde.diagonal()
        
        p = self.dropout_p
        if p >= 1.0: p = 0.99
        w = (p / (1.0 - p + self.eps)) * g_tilde_diag
        
        # Step 5: Solve (G_tilde + diag(w + lambda)) B = G_tilde
        A = G_tilde.clone()
        A.diagonal().add_(w + self.reg_lambda)
        
        self.weight_matrix = torch.linalg.solve(A, G_tilde)
        print("EnergyDLAE fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense()
        user_vec = self.train_matrix_dense[user_indices]
        return user_vec @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
