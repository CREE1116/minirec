import torch
from .base import BaseModel


class ConstrainedMCAREASE(BaseModel):
    """
    Constrained MCAR-EASE (Covariate Balanced EASE)

    철학:
        diag(W)=0 제약 없이 열합 제약만으로
        자연스럽게 대각을 억제하면서 노출 균등화.

        Σ_i W_ij = c_j  where c_j ∝ n_j^γ

        γ=0: c_j=1  → 완전 균등 노출 (MCAR 목표)
        γ=1: c_j∝n_j → EASE에 가까움
        γ∈(0,1): 두 극단 보간

    KKT 유도:
        L = ‖X - XW‖²_F + λ‖W‖²_F + Σ_j ν_j(Σ_i W_ij - c_j)

        ∂L/∂w_j = 0:
            w_j = PG_j - ν_j s     (s = P1)

        열합 제약:
            1^T w_j = c_j
            (1 - λs_j) - ν_j S = c_j
            ν_j = (1 - λs_j - c_j) / S

    Closed-form (1×1, 2×2 불필요):
        ν_j = (1 - λs_j - c_j) / S
        W = (I - λP) - outer(s, ν)

    diag(W)가 0에 가까우면 열합 제약이 자연스럽게 작동 중.
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.lambda_reg = config['model'].get('reg_lambda', 200.0)
        self.gamma      = config['model'].get('gamma', 0.5)
        self.eps        = 1e-12
        self.weight_matrix = None
        self.train_matrix  = None

    def fit(self, data_loader):
        print(f"Fitting ConstrainedMCAREASE "
              f"(lambda={self.lambda_reg}, gamma={self.gamma}) "
              f"on {self.device}...")
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X
        X_dense   = X.to_dense().to(self.device)
        num_items = X_dense.size(1)

        # ── 1. G, P ───────────────────────────────────────────────────────
        G = X_dense.t() @ X_dense
        A = G + self.lambda_reg * torch.eye(num_items, device=self.device)

        try:
            P = torch.linalg.inv(A)
        except (torch._C._LinAlgError, RuntimeError):
            print("[Warning] Singular, stronger regularization.")
            A.diagonal().add_(self.lambda_reg * 10 + 1e-4)
            P = torch.linalg.inv(A)

        # ── 2. 상수 ───────────────────────────────────────────────────────
        s        = P.sum(dim=0)          # P1,  shape [K]
        S_scalar = s.sum()               # 1^T P 1, scalar

        # ── 3. c_j ────────────────────────────────────────────────────────
        n_j   = X_dense.sum(dim=0)
        c_raw = torch.pow(n_j + self.eps, self.gamma)
        c_j   = c_raw * (num_items / (c_raw.sum() + self.eps))

        # ── 4. ν_j closed-form ────────────────────────────────────────────
        nu_sol = (1.0 - self.lambda_reg * s - c_j) / (S_scalar + self.eps)

        # ── 5. W = (I - λP) - outer(s, ν) ────────────────────────────────
        PG = torch.eye(num_items, device=self.device) - self.lambda_reg * P
        self.weight_matrix = PG - torch.outer(s, nu_sol)

        # 대각 억제 확인
        diag_mean = self.weight_matrix.diagonal().abs().mean().item()
        print(f"  mean |diag(W)| = {diag_mean:.6f}")
        print("ConstrainedMCAREASE fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = \
                self.train_matrix.to_dense().to(self.device)
        return self.train_matrix_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None