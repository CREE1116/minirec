import torch
from .base import BaseModel


class LR_Chebyshev_CF(BaseModel):
    """
    LR-Chebyshev-CF: Loop-Removal Chebyshev Collaborative Filtering
    
    Gram 행렬에서 루프 성분(인기도 기반 Rank-1 행렬)을 명시적으로 제거하고,
    잔차 신호를 체비쇼프 다항식 필터로 추출하여 추천 가중치를 학습합니다.
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.K = config['model'].get('K', 5)
        self.tau = config['model'].get('tau', 0.5)  # loop removal strength
        self.eps = 1e-8

        self.weight_matrix = None
        self.train_matrix = None

    def _estimate_max_eigenvalue(self, R, n_iter=10):
        """Power iteration to estimate the largest eigenvalue (Spectral Norm)."""
        v = torch.randn(R.shape[0], device=R.device)
        v = v / v.norm()
        mu = 1.0
        for _ in range(n_iter):
            v = torch.mv(R, v)
            mu = v.norm()
            v = v / (mu + self.eps)
        return float(mu)

    def chebyshev_filter(self, X, R):
        """체비쇼프 필터를 통한 신호 정제"""
        N = R.size(0)

        # Estimate lambda_max using power iteration (MPS compatible)
        lambda_max = self._estimate_max_eigenvalue(R)
        R_tilde = (2.0 / (lambda_max + self.eps)) * R - torch.eye(N, device=R.device)

        T0 = X
        T1 = torch.mm(R_tilde, X)

        out = 0.5 * T0 + 0.5 * T1

        for k in range(2, self.K):
            Tk = 2 * torch.mm(R_tilde, T1) - T0

            # simple decaying coefficient for low-pass effect
            coeff = 1.0 / (k + 1)

            out += coeff * Tk
            T0, T1 = T1, Tk

        return out

    def fit(self, data_loader):
        print(f"Fitting LR-Chebyshev-CF (K={self.K}, tau={self.tau}, lambda={self.reg_lambda}) on {self.device}...")

        X = self.get_train_matrix(data_loader)
        self.train_matrix = X
        X_dense = X.to_dense().to(self.device)

        # ── 1. Gram matrix ────────────────────────────────────────────────
        G = X_dense.t() @ X_dense

        # ── 2. Loop component (rank-1) ────────────────────────────────────
        p = G.sum(dim=1, keepdim=True)  # (I, 1)
        L = p @ p.t()
        # Use Frobenius norm for stable normalization across all devices
        L = L / (torch.norm(L, p='fro') + self.eps)

        # ── 3. Residual Signal (Loop removed) ─────────────────────────────
        R = G - self.tau * L

        # ── 4. Chebyshev filtering ────────────────────────────────────────
        X_filt = self.chebyshev_filter(X_dense.t(), R).t()

        # ── 5. EASE on filtered signal ────────────────────────────────────
        G_filt = X_filt.t() @ X_filt

        A = G_filt.clone()
        A.diagonal().add_(self.reg_lambda)

        try:
            P = torch.linalg.inv(A)
        except RuntimeError:
            print("[Warning] Singular matrix, applying fallback regularization.")
            A.diagonal().add_(self.reg_lambda * 10 + 1e-4)
            P = torch.linalg.inv(A)

        diag_P = P.diagonal()
        W = P / (-diag_P + self.eps)
        W.diagonal().zero_()

        self.weight_matrix = W

        print("LR-Chebyshev-CF fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)
        return self.train_matrix_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
