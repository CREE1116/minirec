
import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy

class CausalAspire(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 10.0) 
        # 분산 안정화 변환(VST)의 수학적 최적점 gamma = 0.5 선언
        self.gamma      = config['model'].get('alpha', 0.5) 
        self.eps        = 1e-12
        self.weight_matrix = None
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting Causal ASPIRE (Standardization Version, gamma={self.gamma})...")

        # 1. Load data
        self.train_matrix_scipy = get_train_matrix_scipy(data_loader)
        X_sp = self.train_matrix_scipy # (U, K)

        # ── Step 1: User-side Propensity (q_u) ───────────────────────────
        # 유저별 활동량(Degree) 추출
        n_u = np.asarray(X_sp.sum(axis=1)).ravel()
        # 분산 안정화를 위한 유저 스케일링: D_U^{-0.5}
        user_weights = np.power(n_u + self.eps, -self.gamma)
        D_U_inv_half = sp.diags(user_weights)

        # ── Step 2: Item-side Bias (b_i) via Purified Expectation ─────────
        # 유저 편향이 1차 제거된 공간에서의 아이템 인기도(Standardized Popularity)
        # Gram matrix의 기댓값 구조인 X.T @ D_U^{-1} @ X 를 효율적으로 계산
        print("  step 2: computing item-side standardized bias...")
        X_u_purified = D_U_inv_half @ X_sp 

        # 아이템별 표준화된 인기도 b_i 추출 (Gram 행렬의 대각 성분)
        # b_i = sum_u (X_ui^2 / q_u^0.5)
        item_bias = np.asarray(X_u_purified.power(2).sum(axis=0)).ravel()

        # ── Step 3: Gram Matrix Standardization (The VST Miracle) ──────────
        # 그램 행렬의 각 원소를 표준편차(sqrt(b_i * b_j))로 나누어 SNR 평탄화
        # G_tilde = D_I^{-0.5} @ (X.T @ D_U^{-0.5} @ X) @ D_I^{-0.5}
        print("  step 3: standardizing gram matrix to equalize SNR...")
        item_weights = np.power(item_bias + self.eps, -self.gamma / 2.0)
        D_I_inv_fourth = sp.diags(item_weights)

        # 최종 정제된 그램 행렬 (CPU Sparse -> Dense)
        # 이 행렬은 모든 아이템 쌍에 대해 균일한 분산을 가짐
        G_U = (X_sp.T @ X_u_purified).toarray() # (X.T @ D_U^{-0.5} @ X)
        G_tilde = D_I_inv_fourth @ G_U @ D_I_inv_fourth

        # ── Step 4: Strict EASE Solution (CPU) ──────────────────────────
        print("  step 4: solving strict EASE (CPU)...")
        # P = (G_tilde + lambda*I)^{-1}
        G_tilde[np.diag_indices_from(G_tilde)] += self.reg_lambda
        P_np = np.linalg.inv(G_tilde)

        # W_{ij} = - P_{ij} / P_{jj} (i != j), W_{ii} = 0
        # EASE의 최적성을 복원하는 해석적 해
        P_diag = np.diag(P_np)
        W_np = -P_np / (P_diag[np.newaxis, :] + self.eps)
        np.fill_diagonal(W_np, 0)

        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        self.train_matrix_gpu = self.get_train_matrix(data_loader)
        print("Causal ASPIRE (Standardization) fitting complete.")

    def forward(self, user_indices):
        input_tensor = torch.index_select(self.train_matrix_gpu, 0, user_indices).to_dense()
        return input_tensor @ self.weight_matrix