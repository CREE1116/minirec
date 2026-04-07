import torch
from .base import BaseModel


class Aspire(BaseModel):
    """
    ASPIRE: Activity-Corrected Exposure (ACE) 기반 스펙트럴 정규화

    ACE_i = n_i² / S_i
          ∝ b_i  (유저 활동량 E[q] 소거 → 순수 노출 편향 추정)

    where:
        n_i = G_ii = Σ_u X_ui          (아이템 자기 에너지)
        S_i = Σ_j G_ij = Σ_u X_ui·n_u  (공출현 에너지 총합)

    Log-space 정규화:
        log(ACE_i) = 2·log(n_i) - log(S_i)
        ACE_i = exp(log(ACE_i) - mean(log(ACE_i)))  ← geometric mean 기준
        → sparse item의 극단값 억제

    학습:
        G_tilde = D^{-α/2} G D^{-α/2}   (ACE-corrected similarity)
        W = EASE(G_tilde)                 (W 내부에 보정 흡수)

    추론:
        ŷ = X @ W                         (D^{-α/2} 재적용 없음 → 과보정 방지)
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.alpha      = config['model'].get('alpha', 1.0)
        self.eps        = 1e-12
        self.weight_matrix = None
        self.train_matrix  = None

    def fit(self, data_loader):
        print(f"Fitting Aspire (lambda={self.reg_lambda}, alpha={self.alpha}) "
              f"on {self.device}...")
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X

        # ── 1. Gram matrix G = X^T X ──────────────────────────────────────
        G = torch.sparse.mm(X.t(), X.to_dense()).to(self.device)
        n_i = G.diagonal()     # 아이템 자기 에너지 (≈ popularity)
        S_i = G.sum(dim=1)     # 공출현 에너지 총합

        # ── 2. ACE: log-space robust estimation ───────────────────────────
        # log(ACE_i) = 2·log(n_i) - log(S_i)
        # geometric mean 기준 중심화 → sparse 극단값 억제
        log_ace = 2.0 * torch.log(n_i + self.eps) \
                -       torch.log(S_i + self.eps)
        ace = (log_ace - log_ace.mean()).exp()

        # ── 3. ACE-corrected Gram ─────────────────────────────────────────
        # G_tilde_ij = G_ij / (ACE_i · ACE_j)^{α/2}
        #            ≈ G_ij / (b_i · b_j)^{α/2}  (편향 보정)
        scale = torch.pow(ace + self.eps, -self.alpha / 2.0)
        G_tilde = G * scale.unsqueeze(1) * scale.unsqueeze(0)

        # ── 4. EASE: (G_tilde + λI)^{-1} ─────────────────────────────────
        A = G_tilde.clone()
        A.diagonal().add_(self.reg_lambda)

        try:
            P = torch.linalg.inv(A)
        except (torch._C._LinAlgError, RuntimeError):
            print("[Warning] Singular matrix, stronger regularization applied.")
            A.diagonal().add_(self.reg_lambda * 10 + 1e-4)
            P = torch.linalg.inv(A)

        # ── 5. W = I - P / diag(P)  (diag(W) = 0) ────────────────────────
        diag_P = P.diagonal()
        self.weight_matrix = P / (-diag_P + self.eps)
        self.weight_matrix.diagonal().zero_()

        print("Aspire fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)
        # ŷ = X @ W
        # D^{-α/2} 재적용 없음: W가 이미 ACE-corrected 공간 기준으로 학습됨
        return self.train_matrix_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None