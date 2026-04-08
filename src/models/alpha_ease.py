import torch
from .base import BaseModel


class AlphaEASE(BaseModel):
    """
    Alpha-EASE: Self-to-Neighbor Energy Ratio Constraint

    제약:
        W_ii = α · Σ_{j≠i} W_ij   ∀i

        α=0:   W_ii=0      → EASE
        α=0.5: 자기:이웃 = 1:3
        α=1:   자기:이웃 = 1:2  (50:50 에너지)
        α→∞:   제약 없음   → Wiener

    자기 에너지 비율:
        W_ii / row_sum_i = α / (1+α)

    KKT 유도:
        L = ‖X-XW‖²_F + λ‖W‖²_F + Σ_i μ_i(W_ii - α·Σ_{j≠i} W_ij)

        ∂L/∂W_ij=0:
            j≠i: (G+λI)W_{ij} = G_{ij} + α·μ_i/2
            j=i: (G+λI)W_{ii} = G_{ii} - μ_i/2

        행렬로:
            (G+λI)W = G + Δ
            Δ_{ij} = α·μ_i/2  (j≠i)
            Δ_{ii} = -μ_i/2

        W = P(G + Δ)
          = PG + P·Δ

        제약 W_ii = α·Σ_{j≠i} W_ij 에서 μ_i 결정:

            (PG)_{ii} - μ_i/2·P_{ii} + α·μ_i/2·(s_i - P_{ii})
            = α·[(PG 행합)_i - (PG)_{ii}
                + α·μ_i/2·(s_i·K̃ - P_{ii}) - μ_i/2·(s_i - P_{ii})]

        간소화 (1×1 방정식):
            μ_i = 2·((PG)_{ii} - α·rowsum_i(PG)) /
                  ((1+α)·P_{ii} - α·s_i·(1+α))
                = 2·((PG)_{ii} - α·(PG·1)_i) /
                  ((1+α)·(P_{ii} - α·s_i))

        W = PG - diag(μ/2) + α·(μ/2)·outer(1,1) 보정
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.alpha      = config['model'].get('alpha', 0.0)  # 0=EASE
        self.eps        = 1e-12
        self.weight_matrix = None
        self.train_matrix  = None

    def fit(self, data_loader):
        print(f"Fitting AlphaEASE "
              f"(lambda={self.reg_lambda}, alpha={self.alpha}) "
              f"on {self.device}...")
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X

        # ── 1. G, P ───────────────────────────────────────────────────────
        G = torch.sparse.mm(X.t(), X.to_dense()).to(self.device)
        K = G.shape[0]
        A = G + self.reg_lambda * torch.eye(K, device=self.device)

        try:
            P = torch.linalg.inv(A)
        except (torch._C._LinAlgError, RuntimeError):
            print("[Warning] Singular, stronger regularization.")
            A.diagonal().add_(self.reg_lambda * 10 + 1e-4)
            P = torch.linalg.inv(A)

        # ── 2. 기본 EASE: W0 = PG = I - λP ───────────────────────────────
        W0 = torch.eye(K, device=self.device) - self.reg_lambda * P  # PG

        # ── 3. α=0이면 EASE로 ─────────────────────────────────────────────
        if abs(self.alpha) < 1e-8:
            diag_P = P.diagonal()
            W = P / (-diag_P + self.eps)
            W.diagonal().zero_()
            self.weight_matrix = W
            print("AlphaEASE (α=0) = EASE complete.")
            return

        # ── 4. μ_i 계산 ───────────────────────────────────────────────────
        # W0 = PG
        # W0_ii:       PG의 대각
        # rowsum(W0)_i: PG의 행합

        W0_diag    = W0.diagonal()                   # (PG)_{ii}
        W0_rowsum  = W0.sum(dim=1)                   # Σ_j (PG)_{ij}

        # 제약: W_ii = α · Σ_{j≠i} W_ij
        # → W_ii = α · (rowsum - W_ii)
        # → (1+α)·W_ii = α · rowsum
        #
        # W = PG + P·Δ 에서:
        #   W_ii     = W0_ii     - μ_i/2·P_ii    + α·μ_i/2·(s_i - P_ii)
        #   rowsum_i = W0_rowsum - μ_i/2·s_i     + α·μ_i/2·(S - s_i)
        #
        # 제약 대입 후 μ_i:

        s        = P.sum(dim=0)                       # P1
        S_scalar = s.sum()                            # 1^T P 1
        P_diag   = P.diagonal()

        # W_ii = W0_ii - μ/2·P_ii + α·μ/2·(s - P_ii)
        #      = W0_ii + μ/2·(-P_ii + α·s - α·P_ii)
        #      = W0_ii + μ/2·(α·s - (1+α)·P_ii)

        # rowsum = W0_rowsum - μ/2·s + α·μ/2·(S - s)
        #        = W0_rowsum + μ/2·(-s + α·S - α·s)
        #        = W0_rowsum + μ/2·(α·S - (1+α)·s)

        # 제약: (1+α)·W_ii = α·rowsum
        # (1+α)·[W0_ii + μ/2·(α·s-(1+α)·P_ii)]
        # = α·[W0_rowsum + μ/2·(α·S-(1+α)·s)]

        # μ/2 항 정리:
        # lhs_coeff = (1+α)·(α·s - (1+α)·P_ii)
        # rhs_coeff = α·(α·S - (1+α)·s)
        # const     = α·W0_rowsum - (1+α)·W0_ii

        lhs_coeff = (1 + self.alpha) * (self.alpha * s - (1+self.alpha) * P_diag)
        rhs_coeff = self.alpha * (self.alpha * S_scalar - (1+self.alpha) * s)
        const     = self.alpha * W0_rowsum - (1+self.alpha) * W0_diag

        # μ/2 = const / (lhs_coeff - rhs_coeff)
        denom  = lhs_coeff - rhs_coeff
        mu_half = const / (denom + self.eps)

        # ── 5. W = W0 + P·Δ ───────────────────────────────────────────────
        # Δ_{ij} = α·μ_i/2  (j≠i)
        # Δ_{ii} = -μ_i/2
        #
        # P·Δ = α·outer(P_mu, 1) - (1+α)·P * mu_half.unsqueeze(1)
        # where P_mu = P @ mu_half

        P_mu = P @ mu_half                            # P·(μ/2)
        W = W0 \
            + self.alpha * torch.outer(P_mu, torch.ones(K, device=self.device)) \
            - (1 + self.alpha) * P * mu_half.unsqueeze(1)

        # ── 6. 검증 ───────────────────────────────────────────────────────
        W_diag    = W.diagonal()
        W_rowsum  = W.sum(dim=1)
        W_offdiag = W_rowsum - W_diag

        ratio     = (W_diag / (W_offdiag + self.eps)).abs()
        ratio_err = (ratio - self.alpha).abs().mean().item()

        print(f"  mean |W_ii/off_diag - α| = {ratio_err:.6f}  "
              f"(0이면 제약 완벽히 만족)")
        print(f"  mean |diag(W)| = {W_diag.abs().mean().item():.6f}")

        self.weight_matrix = W
        print("AlphaEASE fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = \
                self.train_matrix.to_dense().to(self.device)
        return self.train_matrix_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
