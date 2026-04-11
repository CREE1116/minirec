import torch
import torch.nn as nn
from .base import BaseModel


class DRLAE(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        
        # 모델 파라미터 로드 (config['model'] 에서 가져옴)
        model_cfg = config.get('model', {})
        self.lambda_base = float(model_cfg.get('reg_lambda', 500.0))
        self.lambda_var = float(model_cfg.get('lambda_var', 1.0))
        self.min_prop = float(model_cfg.get('min_prop', 1e-4))
        
        self.W = None
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting DRLAE (lambda={self.lambda_base}, var={self.lambda_var} on {self.device}...")
        X_sparse = self.get_train_matrix(data_loader)
        self.train_matrix = X_sparse
        
        # 1. Propensity Score (p_i)
        # torch.sparse.sum can be tricky on MPS, to_dense first if needed or use CPU
        item_counts = torch.sparse.sum(X_sparse, dim=0).to_dense().to(self.device)
        p_vec = (item_counts / (item_counts.max() + 1e-12)) ** self.gamma
        p_vec = torch.clamp(p_vec, min=self.min_prop)
        
        # 2. 공분산 S = X^T X
        X_dense = X_sparse.to_dense().to(self.device)
        S = torch.mm(X_dense.t(), X_dense)
        
        # 3. 편향 제거 (IPS 방식): S_star = S / (p_i * p_j)
        # Memory efficient way to compute S / outer(p, p)
        S_star = S / p_vec.unsqueeze(0)
        S_star = S_star / p_vec.unsqueeze(1)
        
        # 4. Variance-Penalized 정규화 (Omega)
        S_diag = torch.diag(S)
        var_penalty = ((1.0 - p_vec**2) / (p_vec**4 + 1e-12)) * (S_diag**2)
        omega_diag = self.lambda_base + self.lambda_var * var_penalty
        
        # 5. 역행렬 연산
        A = S_star.clone()
        A.diagonal().add_(omega_diag)
        
        print("Solving linear system...")
        try:
            A_inv = torch.linalg.inv(A)
        except (torch._C._LinAlgError, RuntimeError):
            print("[Warning] Singular matrix, adding extra regularization.")
            A.diagonal().add_(self.lambda_base * 10)
            A_inv = torch.linalg.inv(A)
        
        # 6. 최종 LAE 가중치 W 도출
        P_inv_diag = torch.diag(A_inv)
        self.W = - A_inv / (P_inv_diag.unsqueeze(0) + 1e-12)
        self.W.fill_diagonal_(0.0)
        
        print("DRLAE fitting complete.")

    def forward(self, user_indices):
        if self.W is None:
            raise RuntimeError("Model must be fitted before prediction.")
            
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)
            
        X_users = self.train_matrix_dense[user_indices]
        
        # 스코어 계산: X_u @ W
        scores = torch.mm(X_users, self.W)
        
        return scores
    
    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
