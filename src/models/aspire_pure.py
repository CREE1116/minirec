import torch
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix
import torch
import numpy as np

class AspirePure(BaseModel):
    """
    ASPIRE-DAE: Topological Denoising Linear Autoencoder
    - Removes hard diag(W)=0 constraint.
    - Uses generalized topological penalty D_i as dropout-like regularization.
    - Extremely elegant closed-form: W = I - \lambda * P * D
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        
        # 위상적 페널티 밸런스 (0.0: IPS, 0.5~1.0: Hub Penalty)
        self.gamma      = config['model'].get('gamma', 0.5) 
        self.eps        = 1e-12
        self.weight_matrix = None
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting ASPIRE-DAE (lambda={self.reg_lambda}, gamma={self.gamma}) on {self.device}...")
        
        X = get_train_matrix_scipy(data_loader)
        self.train_matrix_scipy = X

        # ── 1. Gram matrix G = X^T X (CPU -> GPU) ───────────────────────────
        G_np = compute_gram_matrix(X)
        G = torch.tensor(G_np, dtype=torch.float32, device=self.device)
        
        d = G.diagonal()       
        S = G.sum(dim=1)       

        d_smooth = d + 1.0
        S_smooth = S + 1.0  
        
        # ── 2. Topological Penalty D (DLAE의 Dropout 분산 역할을 대체) ───────
        # D_i = d_i^(1-gamma) * S_i^gamma
        log_D = (1.0 - self.gamma) * torch.log(d_smooth) + self.gamma * torch.log(S_smooth)
        log_D_centered = log_D - log_D.mean()
        
        # 대각 규제 행렬 D 벡터
        D_vector = torch.exp(log_D_centered)

        # ── 3. DAE Closed-form Inversion ────────────────────────────────────
        # A = G + \lambda * D
        A = G.clone()
        A.diagonal().add_(self.reg_lambda * D_vector)
        
        print(f"  inverting matrix on {self.device}...")
        P = torch.linalg.inv(A)
        
        # 🚀 [크리의 통찰이 만든 우아한 수식] W = I - \lambda * P * D
        # 기존 EASE 방식의 나눗셈(-P_ij / P_ii)과 하드 제약을 완전히 제거
        W = -self.reg_lambda * P * D_vector.view(1, -1)
        W.diagonal().add_(1.0) # I 더하기
        
        # (선택) 추론 시 이미 소비한 아이템 추천을 막기 위해 대각을 0으로 칠해도 되고 안해도 됨.
        # 보수적으로 0으로 칠해두는 것이 랭킹 메트릭(NDCG) 측정 시 안전함.
        W.diagonal().zero_() 

        self.weight_matrix = W
        print("ASPIRE-DAE fitting complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_matrix = self.train_matrix_scipy[users].toarray()
        input_tensor = torch.tensor(input_matrix, dtype=torch.float32, device=self.device)
        
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None