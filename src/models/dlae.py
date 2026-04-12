import torch
import torch.nn as nn
import numpy as np
from scipy import sparse
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class DLAE(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.dropout_p = config['model'].get('dropout_p', 0.5)
        self.weight_matrix = None
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting DLAE (p={self.dropout_p}, lambda={self.reg_lambda}) on {self.device}...")
        
        X = get_train_matrix_scipy(data_loader)
        self.train_matrix_scipy = X

        print("  computing gram matrix (CPU)...")
        G_np = compute_gram_matrix(X)
        
        # Perform calculations on CPU to avoid VRAM issues
        p = min(self.dropout_p, 0.99)
        w = (p / (1.0 - p)) * G_np.diagonal()

        G_lhs = G_np.copy()
        G_lhs[np.diag_indices_from(G_lhs)] += (w + self.reg_lambda)
        
        print("  solving linear system (CPU)...")
        W_np = np.linalg.solve(G_lhs, G_np)
        
        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        print("DLAE fitting complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_matrix = self.train_matrix_scipy[users].toarray()
        input_tensor = torch.tensor(input_matrix, dtype=torch.float32, device=self.device)
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
