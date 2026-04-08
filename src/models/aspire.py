import torch
from .base import BaseModel


class Aspire(BaseModel):
    """
    ASPIRE: Spectral Purification via Signal-to-Noise Ratio (d^2/S)
    implemented with standard Ridge Regression (Diagonal enabled).

    SNR을 로그 공간에서 추정 (geometric mean 기준 정규화):
        log(λ_i) = 2·log(n_i) - log(S_i)
        λ_i_robust = exp(log(λ_i) - mean(log(λ_i)))
                   = (n_i²/S_i) / GM(n_i²/S_i)

    1. λ_i = exp(2·log(n_i) - log(S_i) - μ)  (log-space robust SNR)
    2. G_tilde = Λ^{-α/2} G Λ^{-α/2}
    3. W = (G_tilde + λI)^{-1} G_tilde  (Standard Ridge Regression)
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.alpha      = config['model'].get('alpha', 1.0)
        self.eps        = 1e-12
        self.weight_matrix = None
        self.train_matrix  = None

    def fit(self, data_loader):
        print(f"Fitting Aspire (Standard Ridge, lambda={self.reg_lambda}, alpha={self.alpha}) "
              f"on {self.device}...")
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X

        # ── 1. Gram matrix G = X^T X ──────────────────────────────────────
        G = torch.sparse.mm(X.t(), X.to_dense()).to(self.device)
        d = G.diagonal()       # n_i: 아이템 자기 에너지 (≈ popularity)
        S = G.sum(dim=1)       # S_i: 그램 행합 (공출현 에너지)

        # ── 2. Log-space robust SNR ────────────────────────────────────────
        log_lambda = 2.0 * torch.log(d + self.eps) - torch.log(S + self.eps)
        log_lambda_centered = log_lambda - log_lambda.mean()
        reliability = log_lambda_centered.exp()

        # ── 3. G_tilde = Λ^{-α/2} G Λ^{-α/2} ────────────────────────────
        scale_factor = torch.pow(reliability + self.eps, -self.alpha / 2.0)
        G_tilde = G * scale_factor.unsqueeze(1) * scale_factor.unsqueeze(0)

        # ── 4. Standard Ridge Regression (Diagonal enabled) ──────────────
        A = G_tilde.clone()
        A.diagonal().add_(self.reg_lambda)

        # W = (G_tilde + λI)^{-1} G_tilde
        # linalg.solve가 inv보다 수치적으로 더 안정적입니다.
        self.weight_matrix = torch.linalg.solve(A, G_tilde)

        print("Aspire fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)
        return self.train_matrix_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
