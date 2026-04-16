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
        print(f"Fitting EASE (lambda={self.reg_lambda}) on CPU...")
        
        # Use shared utility for efficient sparse matrix loading
        self.train_matrix_scipy = get_train_matrix_scipy(data_loader)
        X = self.train_matrix_scipy

        print("  computing gram matrix (CPU)...")
        G_np = compute_gram_matrix(X, data_loader)
        
        # Move to CPU for inversion to avoid VRAM issues
        print("  inverting matrix (CPU)...")
        G_np = G_np.copy()
        G_np[np.diag_indices_from(G_np)] += self.reg_lambda
        
        P_np = np.linalg.inv(G_np)
        
        # Final weights: B_{ij} = -P_{ij} / P_{jj} (i != j), B_{ii} = 0
        diag_P = np.diag(P_np)
        B_np = -P_np / (diag_P + 1e-12)
        np.fill_diagonal(B_np, 0)
        
        self.weight_matrix = torch.tensor(B_np, dtype=torch.float32, device=self.device)
        
        # GPU Sparse Tensor 준비 (forward 가속)
        print("  Preparing GPU Sparse Matrix for fast inference...")
        self.train_matrix_gpu = self.get_train_matrix(data_loader)
        
        print("EASE fitting complete.")

    def forward(self, user_indices):
        # GPU Sparse Tensor에서 행 선택 후 dense 변환하여 행렬곱 (초고속)
        input_tensor = torch.index_select(self.train_matrix_gpu, 0, user_indices).to_dense()
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
