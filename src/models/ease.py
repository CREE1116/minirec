import torch
import torch.nn as nn
import numpy as np
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class EASE(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.weight_matrix = None
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting EASE (lambda={self.reg_lambda}) on {self.device}...")
        
        # Use shared utility for efficient sparse matrix loading
        self.train_matrix_scipy = get_train_matrix_scipy(data_loader)
        X = self.train_matrix_scipy

        print("  computing gram matrix (CPU)...")
        G_np = compute_gram_matrix(X, data_loader)
        
        # Move to GPU for fast inversion
        print(f"  inverting matrix on {self.device}...")
        G = torch.tensor(G_np, dtype=torch.float32, device=self.device)
        G.diagonal().add_(self.reg_lambda)
        
        P = torch.linalg.inv(G)
        
        # Final weights
        B = P / (-P.diagonal().view(-1, 1) + 1e-12)
        B.diagonal().zero_()
        
        self.weight_matrix = B
        print("EASE fitting complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_matrix = self.train_matrix_scipy[users].toarray()
        input_tensor = torch.tensor(input_matrix, dtype=torch.float32, device=self.device)
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
