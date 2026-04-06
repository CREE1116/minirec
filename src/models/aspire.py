import torch
import torch.nn as nn
from .base import BaseModel

class Aspire(BaseModel):
    """
    Pure ASPIRE: Spectral Purification via Signal-to-Noise Ratio (d^2/S).
    Matches the derivation: G_tilde = Lambda^(-alpha/2) * G * Lambda^(-alpha/2)
    where Lambda = diag(d^2 / S).
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.alpha = config['model'].get('alpha', 1.0) # alpha=1.0 is the theoretical starting point
        self.eps = 1e-12
        self.weight_matrix = None
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting Pure Aspire (alpha={self.alpha}, lambda={self.reg_lambda}) on {self.device}...")
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X # Inference 시 사용

        # 1. Gram Matrix 계산 (X.T @ X)
        # 아이템 수가 많을 경우를 대비해 sparse-dense 곱 이후 dense로 변환
        G = torch.sparse.mm(X.t(), X.to_dense()).to(self.device)
        
        # 2. 핵심 지표 계산: d (Popularity), S (Row-sum of G)
        d = G.diagonal()
        S = G.sum(dim=1)

        # 3. 신뢰도 행렬 Lambda (d^2 / S) 계산
        # beta를 제거하고 수치적 안정성을 위해 eps만 추가
        reliability = (d * d) / (S + self.eps)
        
        # 4. Scale Restoration (스케일 복원)
        # alpha 변화에 따른 reg_lambda의 민감도를 낮추기 위해 평균 신뢰도를 1로 정규화
        norm_reliability = reliability / (reliability.mean() + self.eps)

        # 5. 대칭 정규화 (Spectral Purification)
        # G_tilde = Lambda^(-alpha/2) * G * Lambda^(-alpha/2)
        # 부호(-)는 편향(거품)이 큰 아이템의 가중치를 낮추는 정규화 방향
        scale_factor = torch.pow(norm_reliability + self.eps, -self.alpha / 2.0)
        G_tilde = G * scale_factor.unsqueeze(1) * scale_factor.unsqueeze(0)

        # 6. Ridge Regression (Closed-form Solution)
        # Diag-zero 제약 없이 수식(Pure Wiener Filter) 그대로 해결
        A = G_tilde.clone()
        A.diagonal().add_(self.reg_lambda)
        
        # (G_tilde + lambda*I) W = G_tilde 를 만족하는 W 계산
        self.weight_matrix = torch.linalg.solve(A, G_tilde)
        
        print("Pure Aspire fitting complete.")

    def forward(self, user_indices):
        # 학습 행렬과 가중치 행렬의 내적을 통해 예측값 산출
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)
        return self.train_matrix_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        # 선형 모델은 closed-form으로 직접 해를 구하므로 별도의 Loss 학습 불필요
        return (torch.tensor(0.0, device=self.device),), None
