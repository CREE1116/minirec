import torch
import torch.nn as nn
from .base import BaseModel

class Aspire(BaseModel):
    """
    Degree-Ratio Symmetric Wiener Filter with Bayesian Anchored Influence.
    Influence I = d * (d + beta) / (S + beta * mu_global)
    G_tilde = I^(-alpha/2) * G * I^(-alpha/2)
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.alpha = config['model'].get('alpha', 2.0)
        self.beta = config['model'].get('beta', 20.0)
        self.eps = 1e-12
        self.weight_matrix = None
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting Aspire (alpha={self.alpha}, beta={self.beta}, lambda={self.reg_lambda}) on {self.device}...")
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X

        G = torch.sparse.mm(X.t(), X.to_dense()).to(self.device)
        d = G.diagonal()
        S = G.sum(dim=1)

        # 1. Global Prior 계산 (전체 활동량 / 전체 차수)
        mu_global = S.sum() / (d.sum() + self.eps)

        # 2. Bayesian Anchored Influence 계산
        influence = d * ((d + self.beta) / (S + self.beta * mu_global + self.eps))
        
        # 3. Scale Restoration: 인플루언스의 평균값으로 나누어 전체 행렬의 스케일을 보존
        # 이렇게 하면 G_tilde의 대각 성분 스케일이 원본 G와 유사해져서 lambda 값이 안정화됩니다.
        influence = influence / (influence.mean() + self.eps)

        # 4. Apply influence-based normalization
        d_inv = torch.pow(influence + self.eps, -self.alpha / 2.0)
        G_tilde = G * d_inv.unsqueeze(1) * d_inv.unsqueeze(0)

        # 5. Ridge Regression (without diagonal-zero constraint)
        A = G_tilde.clone()
        A.diagonal().add_(self.reg_lambda)
        self.weight_matrix = torch.linalg.solve(A, G_tilde)
        print("Aspire fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)
        return self.train_matrix_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
