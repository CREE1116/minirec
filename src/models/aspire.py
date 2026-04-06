import torch
import torch.nn as nn
from .base import BaseModel

class Aspire(BaseModel):
    """
    ASPIRE with Diagonal-Zero Constraint
    1. Spectral Purification via Reliability Index (d^2/S)
    2. EASE-style Diagonal-Zero Constraint for exact top-K prediction
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.alpha = config['model'].get('alpha', 1.0)
        self.eps = 1e-12
        self.weight_matrix = None
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting Pure ASPIRE+DiagZero (lambda={self.reg_lambda}, alpha={self.alpha}) on {self.device}...")
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X

        # 1. 원본 Gram 행렬 계산
        G = torch.sparse.mm(X.t(), X.to_dense()).to(self.device)
        d = G.diagonal()
        S = G.sum(dim=1)

        # 2. 신뢰도 기반 대칭 정규화 (Spectral Purification)
        reliability = (d * d) / (S + self.eps)
        norm_reliability = reliability / (reliability.mean() + self.eps)
        
        scale_factor = torch.pow(norm_reliability + self.eps, -self.alpha / 2.0)
        G_tilde = G * scale_factor.unsqueeze(1) * scale_factor.unsqueeze(0)

        # 3. 정규 방정식 세팅
        A = G_tilde.clone()
        A.diagonal().add_(self.reg_lambda)

        # 4. Diagonal-Zero Constraint 해법 (Inverse 활용)
        try:
            P = torch.linalg.inv(A)
        except torch._C._LinAlgError:
            # 수치적 극단값 대비 안전장치
            print("Warning: Singular matrix, applying safe fallback regularization.")
            A.diagonal().add_(1.0)
            P = torch.linalg.inv(A)

        # W = I - P * diag(P)^-1 (대각 성분 0으로 강제)
        diag_P = P.diagonal()
        self.weight_matrix = P / -diag_P.unsqueeze(0)
        self.weight_matrix.diagonal().fill_(0)
        
        print("Pure ASPIRE fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)
        return self.train_matrix_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None