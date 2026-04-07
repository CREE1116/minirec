import torch
from .base import BaseModel


class AspireReg(BaseModel):
    """
    ASPIRE-Reg: ACE-based Adaptive Regularization

    DLAE와 동일한 구조:
        W = (G + diag(w_i + λ))^{-1} G

    차이점:
        DLAE: w_i = p/(1-p) · n_i          (인기도 비례)
        ASPIRE-Reg: w_i = p/(1-p) · ACE_i  (편향 지수 비례)

    ACE_i = n_i² / S_i  (Activity-Corrected Exposure)
          ∝ b_i  (E[q] 소거된 순수 노출 편향)

    스케일 정합:
        ACE를 log-space에서 geometric mean 기준 정규화
        → n_i와 동일한 스케일로 맞춤
        → p의 의미가 DLAE와 동일하게 유지됨

    대각 제약 없음 (DLAE와 동일):
        dropout 앙상블 관점에서 불필요
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.dropout_p  = config['model'].get('dropout_p', 0.5)
        self.eps        = 1e-12
        self.weight_matrix = None
        self.train_matrix  = None

    def fit(self, data_loader):
        print(f"Fitting AspireReg (lambda={self.reg_lambda}, p={self.dropout_p}) "
              f"on {self.device}...")
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X

        # ── 1. Gram matrix ────────────────────────────────────────────────
        G = torch.sparse.mm(X.t(), X.to_dense()).to(self.device)
        n_i = G.diagonal()       # 아이템 자기 에너지 ≈ popularity
        S_i = G.sum(dim=1)       # 그램 행합 = 공출현 에너지

        # ── 2. ACE: log-space 정규화 ──────────────────────────────────────
        # raw ACE = n_i²/S_i ∝ b_i  (E[q] 소거)
        # log-space geometric mean 기준 중심화
        # → DLAE의 n_i와 동일한 스케일로 정합
        log_ace = 2.0 * torch.log(n_i + self.eps) \
                -       torch.log(S_i + self.eps)
        log_ace_centered = log_ace - log_ace.mean()
        ace = log_ace_centered.exp()                # geometric mean = 1

        # n_i의 mean 스케일로 맞춤 (p의 의미를 DLAE와 통일)
        ace = ace * n_i.mean()

        # ── 3. 적응형 규제: DLAE 구조 그대로, n_i → ACE ──────────────────
        # DLAE: w_i = p/(1-p) · n_i
        # ASPIRE-Reg: w_i = p/(1-p) · ACE_i
        p = min(self.dropout_p, 0.99)
        w = (p / (1.0 - p)) * ace

        A = G.clone()
        A.diagonal().add_(w + self.reg_lambda)

        # ── 4. 정규 방정식 ────────────────────────────────────────────────
        # W = (G + diag(w + λ))^{-1} G
        self.weight_matrix = torch.linalg.solve(A, G)

        print("AspireReg fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)
        return self.train_matrix_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None