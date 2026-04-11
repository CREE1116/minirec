import torch
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix
import torch
import numpy as np
import scipy.sparse as sp


class AspirePure(BaseModel):
    """
    Causal ASPIRE: Two-stage Causal Normalization for Linear CF
    - Now with EASE-style Diagonal Constraint (diag(W)=0)
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 200.0)
        self.alpha      = config['model'].get('alpha', 1.0)
        self.eps        = 1e-12
        self.weight_matrix = None
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting AspirePure (lambda={self.reg_lambda}, alpha={self.alpha}) on {self.device}...")

        # Optimized sparse matrix loading
        self.train_matrix_scipy = get_train_matrix_scipy(data_loader)
        X_sp = self.train_matrix_scipy

        n_u = np.asarray(X_sp.sum(axis=1)).ravel()  # (U,)
        K   = X_sp.shape[1]

        # ── Stage 1: User-side IPW (CPU Sparse) ───────────────────────────
        print("  stage 1: computing user-side IPW gram matrix...")
        D_U_inv = sp.diags(1.0 / (n_u + self.eps))
        X_weighted = D_U_inv @ X_sp                 
        G_U = (X_sp.T @ X_weighted).toarray()       

        # ── Stage 2: Item-side Geometric Ensemble Normalization ───────────
        print("  stage 2: applying item-side geometric ensemble normalization...")
        A_i = G_U.diagonal().copy()
        
        # alpha는 완벽한 대수학적 기하평균 결합의 가중치
        scale = np.power(A_i + self.eps, -self.alpha / 2.0)
        G_tilde = G_U * scale[:, None] * scale[None, :]

        # 진단: α=1 일 때 대각선이 완벽히 1.0 에 수렴하는지 팩트 체크
        if abs(self.alpha - 1.0) < 1e-6:
            diag_check = G_tilde.diagonal()
            print(f"    [α=1 check] diag mean={diag_check.mean():.4f} (이론적으로 정확히 1.0)")

        # ── Stage 3: Unconstrained Closed-form (GPU) ──────────────────────
        print(f"  stage 3: inverting and applying unconstrained closed-form on {self.device}...")
        G_torch = torch.tensor(G_tilde, dtype=torch.float32, device=self.device)
        
        # 대각 체급이 통일되었으므로 단순 L2 페널티만으로 충분함
        A_mat = G_torch + self.reg_lambda * torch.eye(K, device=self.device)

        try:
            P = torch.linalg.inv(A_mat)
        except (torch._C._LinAlgError, RuntimeError):
            print("[Warning] Singular matrix, applying stronger regularization.")
            A_mat.diagonal().add_(self.reg_lambda * 10 + 1e-4)
            P = torch.linalg.inv(A_mat)

        # [핵심 수정] EASE style diagonal constraint 완벽히 제거! 
        # 순수한 릿지 회귀의 닫힌 해만 사용
        W = -P
        
        # 사후에 자기 자신 추천만 마스킹 처리
        W.diagonal().zero_()

        self.weight_matrix = W
        print("AspirePure fitting complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_matrix = self.train_matrix_scipy[users].toarray()
        input_tensor = torch.tensor(input_matrix, dtype=torch.float32, device=self.device)
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None