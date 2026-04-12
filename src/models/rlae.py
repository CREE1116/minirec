import torch
import numpy as np
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class RLAE(BaseModel):
    """
    Relaxed Linear AutoEncoder (RLAE)
    - Optimization: min ||X - XB||^2 + lambda ||B||^2  s.t.  diag(B) <= b
    - b=0: EASE, b>=1: Standard Ridge (LAE)
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.b = config['model'].get('b', 0.0) # Constraint relaxation parameter
        self.eps = 1e-12
        self.weight_matrix = None
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting RLAE (lambda={self.reg_lambda}, b={self.b}) on CPU...")
        X = get_train_matrix_scipy(data_loader)
        self.train_matrix_scipy = X
        G_np = compute_gram_matrix(X, data_loader)
        
        # CPU에서 행렬 연산 수행
        A = G_np.copy()
        A[np.diag_indices_from(A)] += self.reg_lambda
        
        print(f"  inverting matrix on CPU...")
        P = np.linalg.inv(A)
        
        diag_P = np.diag(P)
        # penalty = lambda + mu = max(lambda, (1-b)/P_jj)
        penalty = np.maximum(self.reg_lambda, (1.0 - self.b) / (diag_P + self.eps))
        
        # W = I - P @ diag(penalty)
        W = - (P * penalty[np.newaxis, :])
        W[np.diag_indices_from(W)] += 1.0
        
        self.weight_matrix = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_matrix_gpu = self.get_train_matrix(data_loader)
        print("RLAE fitting complete.")

    def forward(self, user_indices):
        input_tensor = torch.index_select(self.train_matrix_gpu, 0, user_indices).to_dense()
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None

class RDLAE(BaseModel):
    """
    Relaxed Denoising Linear AutoEncoder (RDLAE)
    - Optimization: min ||X - XB||^2 + ||Lambda^(1/2) B||^2  s.t.  diag(B) <= b
    - Integrates Dropout-like regularization (DLAE style) with relaxed constraints.
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.dropout_p = config['model'].get('dropout_p', 0.5)
        self.b = config['model'].get('b', 0.0)
        self.eps = 1e-12
        self.weight_matrix = None
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting RDLAE (lambda={self.reg_lambda}, p={self.dropout_p}, b={self.b}) on CPU...")
        X = get_train_matrix_scipy(data_loader)
        self.train_matrix_scipy = X
        G_np = compute_gram_matrix(X, data_loader)
        
        # Lambda_jj = (p/(1-p)) * G_jj + lambda
        p = min(self.dropout_p, 0.99)
        dropout_penalty = (p / (1.0 - p)) * np.diag(G_np)
        lambda_diag = dropout_penalty + self.reg_lambda
        
        A = G_np.copy()
        A[np.diag_indices_from(A)] += lambda_diag
        
        print(f"  inverting matrix on CPU...")
        P = np.linalg.inv(A)
        
        diag_P = np.diag(P)
        # total_penalty = lambda_jj + mu_j = max(lambda_jj, (1-b)/P_jj)
        total_penalty = np.maximum(lambda_diag, (1.0 - self.b) / (diag_P + self.eps))
        
        # W = I - P @ diag(total_penalty)
        W = - (P * total_penalty[np.newaxis, :])
        W[np.diag_indices_from(W)] += 1.0
        
        self.weight_matrix = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_matrix_gpu = self.get_train_matrix(data_loader)
        print("RDLAE fitting complete.")

    def forward(self, user_indices):
        input_tensor = torch.index_select(self.train_matrix_gpu, 0, user_indices).to_dense()
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
