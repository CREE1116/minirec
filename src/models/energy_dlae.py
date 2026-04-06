import torch
import numpy as np
import scipy.sparse as sp
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
        self.train_matrix = sp.csr_matrix((self.n_users, self.n_items))

    def fit(self, data_loader):
        print(f"Fitting EnergyDLAE (p={self.dropout_p}, alpha={self.alpha}, lambda={self.reg_lambda})...")
        train_df = data_loader.train_df
        rows, cols = train_df['user_id'].values, train_df['item_id'].values
        X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), 
                          shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix = X
        
        # Step 1: Gram matrix G = X.T @ X
        G = (X.T @ X).toarray()
        
        # Step 2: 2nd-Order Statistic - Row Energy of G
        row_energy = np.sqrt(np.sum(np.square(G), axis=1))
        
        # Step 3: Symmetric Energy Normalization
        e_inv = 1.0 / (np.power(row_energy + self.eps, self.alpha))
        G_tilde = G * e_inv[:, np.newaxis] * e_inv[np.newaxis, :]
        
        # Step 4: DLAE-style Diagonal Regularization using Energy of G_tilde
        g_tilde_diag = np.diag(G_tilde)
        
        p = self.dropout_p
        if p >= 1.0: p = 0.99
        w = (p / (1.0 - p + self.eps)) * g_tilde_diag
        
        # Step 5: Solve (G_tilde + diag(w + lambda)) B = G_tilde
        A = G_tilde.copy()
        np.fill_diagonal(A, g_tilde_diag + w + self.reg_lambda)
        
        B = np.linalg.solve(A, G_tilde)
        
        self.weight_matrix = torch.from_numpy(B).float().to(self.device)
        print("EnergyDLAE fitting complete.")

    def forward(self, user_indices):
        u_ids = user_indices.cpu().numpy()
        user_vec = torch.from_numpy(self.train_matrix[u_ids].toarray()).float().to(self.device)
        return user_vec @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
