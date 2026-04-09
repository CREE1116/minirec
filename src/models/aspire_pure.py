import torch
from .base import BaseModel


class AspirePure(BaseModel):
    """
    ASPIRE-Pure: Pure Shrinkage ACE

    문제:
        ACE_i = n_i² / S_i 는 독립 가정(q_u ⊥ R_ui) 위반 시 폭발
        Steam처럼 특정 게임만 하는 유저가 많으면:
            E[q|R_ui=1] << E[q]  → ACE_i 과대 추정 → 폭발

    해결:
        소비자 평균 활동량(S_i/n_i)에 사전 분포(E[q])를 더해 수축

        D_i = n_i / (S_i/n_i + E[q])
            = n_i² / (S_i + E[q] · n_i)

    수축 동작:
        S_i/n_i >> E[q]: D_i ≈ n_i²/S_i = ACE  (독립 성립 → 그대로)
        S_i/n_i ≈ E[q]:  D_i ≈ n_i/2E[q]       (중간)
        S_i/n_i << E[q]: D_i ≈ n_i/E[q]         (선형, 폭발 방지)

    파라미터:
        reg_lambda: EASE 정규화
        alpha:      보정 강도 [0, 1]
        shrink:     E[q] 가중치 (default=1.0, 높을수록 더 수축)
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.alpha      = config['model'].get('alpha', 1.0)
        self.shrink     = config['model'].get('shrink', 0.1)
        self.eps        = 1e-12
        self.weight_matrix = None
        self.train_matrix  = None
 
    def fit(self, data_loader):
        print(f"Fitting AspireBayes "
              f"(lambda={self.reg_lambda}, alpha={self.alpha}, "
              f"shrink={self.shrink}) on {self.device}...")
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X
 
        # ── 1. Gram matrix ────────────────────────────────────────────────
        G = torch.sparse.mm(X.t(), X.to_dense()).to(self.device)
        n_i = G.diagonal()
        S_i = G.sum(dim=1)
 
        # ── 2. ACE (raw) ──────────────────────────────────────────────────
        ace_raw = n_i ** 2 / (S_i + self.eps)
 
        # ── 3. Bayesian Shrinkage ─────────────────────────────────────────
        # D_i = n_i² / (S_i + shrink * n_i²)
        #     = 1 / (S_i/n_i² + shrink)
        # n_i 클수록 S_i/n_i² 작아져서 shrink가 분모 지배 → 강하게 수축
        D = n_i ** 2 / (S_i + self.shrink * n_i ** 2 + self.eps)
 
        # log-space 중심화
        log_D = torch.log(D + self.eps)
        D = torch.exp(log_D - log_D.mean())
 
        # 진단
        print(f"  ACE kurtosis={self._kurtosis(ace_raw):.2f}, "
              f"max={ace_raw.max().item():.2f}")
        print(f"  D_b kurtosis={self._kurtosis(D):.2f}, "
              f"max={D.max().item():.2f}")
        print(f"  corr(ACE, D_b)={self._corr(ace_raw, D):.4f}")
 
        # ── 4. G_tilde = D^{-α/2} G D^{-α/2} ────────────────────────────
        scale   = torch.pow(D + self.eps, -self.alpha / 2.0)
        G_tilde = G * scale.unsqueeze(1) * scale.unsqueeze(0)
 
        # ── 5. EASE ───────────────────────────────────────────────────────
        A = G_tilde + self.reg_lambda * torch.eye(
                G.shape[0], device=self.device)
 
        try:
            P = torch.linalg.inv(A)
        except (torch._C._LinAlgError, RuntimeError):
            print("[Warning] Singular, stronger regularization.")
            A.diagonal().add_(self.reg_lambda * 10 + 1e-4)
            P = torch.linalg.inv(A)
 
        diag_P = P.diagonal()
        W = P / (-diag_P + self.eps)
        W.diagonal().zero_()
 
        self.weight_matrix = W
        print("AspireBayes fitting complete.")
 
    def _kurtosis(self, x):
        x = x - x.mean()
        return (x**4).mean() / ((x**2).mean()**2 + self.eps) - 3
 
    def _corr(self, x, y):
        x = x - x.mean()
        y = y - y.mean()
        return (x * y).sum() / (
            torch.sqrt((x**2).sum() * (y**2).sum()) + self.eps)
 
    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = \
                self.train_matrix.to_dense().to(self.device)
        return self.train_matrix_dense[user_indices] @ self.weight_matrix
 
    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
 