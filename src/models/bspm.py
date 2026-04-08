import torch
from .base import BaseModel


class BSPM(BaseModel):
    """
    Blurring-Sharpening Process Models for Collaborative Filtering
    SIGIR 2023 - Choi et al.

    논문의 핵심 두 단계:
      1) Blurring  : dZ/dt = (G_b - I) Z   (low-pass → 스무딩)
      2) Sharpening: dZ/dt = (I - G_s) Z   (reverse → 고주파 복원)

    Euler 이산화:
      Blur step   : Z ← Z + h_b (Z @ G_b - Z) = (1 - h_b) Z + h_b (Z @ G_b)
      Sharpen step: Z ← Z + h_s (Z - Z @ G_s) = (1 + h_s) Z - h_s (Z @ G_s)

    하이퍼파라미터 (논문 권장 기본값):
      K_b            : 블러링 step 수          (default: 2)
      T_b            : 블러링 총 시간           (default: 1.0)  → h_b = T_b / K_b
      K_s            : 샤프닝 step 수          (default: 1)
      T_s            : 샤프닝 총 시간           (default: 2.5)  → h_s = T_s / K_s
      idl_beta       : 최종 IDL 샤프닝 계수    (default: 0.2)
      final_sharpening: 최종 한 번 더 샤프닝   (default: True)
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.K_b             = config['model'].get('K_b',              2)
        self.T_b             = config['model'].get('T_b',            1.0)
        self.K_s             = config['model'].get('K_s',              1)
        self.T_s             = config['model'].get('T_s',            2.5)
        self.idl_beta        = config['model'].get('idl_beta',       0.2)
        self.final_sharpening = config['model'].get('final_sharpening', True)

    # ------------------------------------------------------------------
    def fit(self, data_loader):
        print(f"Fitting BSPM  K_b={self.K_b} T_b={self.T_b}  "
              f"K_s={self.K_s} T_s={self.T_s}  "
              f"idl_beta={self.idl_beta}  final_sharpen={self.final_sharpening}")

        R = self.get_train_matrix(data_loader).to(self.device).to_dense()
        self.R = R  # (U, I)

        # ── Blurring kernel G_b : row-stochastic item-item gram ──────
        #    G_b = D^{-1} (R^T R),  eigenvalues ∈ [0, 1]
        G_b = R.t() @ R                                        # (I, I)
        D_b = G_b.sum(dim=1, keepdim=True).clamp(min=1e-12)
        G_b = G_b / D_b

        # ── Sharpening kernel G_s : 논문에서 동일한 low-pass kernel 사용 ──
        #    (sharpening은 high-pass 방향으로 적용하므로 같은 G를 씀)
        G_s = G_b

        # ── 1) Blurring phase  (Euler ODE integration) ───────────────
        #    ODE  : dZ/dt = (G_b - I) Z
        #    Euler: Z_{n+1} = (1 - h_b) Z_n + h_b (Z_n @ G_b)
        h_b = self.T_b / self.K_b
        Z = R.clone()
        for _ in range(self.K_b):
            Z = (1.0 - h_b) * Z + h_b * (Z @ G_b)

        # ── 2) Sharpening phase  (reverse Euler) ─────────────────────
        #    ODE  : dZ/dt = (I - G_s) Z   (reverse direction)
        #    Euler: Z_{n+1} = (1 + h_s) Z_n - h_s (Z_n @ G_s)
        h_s = self.T_s / self.K_s
        for _ in range(self.K_s):
            Z = (1.0 + h_s) * Z - h_s * (Z @ G_s)

        # ── 3) Final IDL sharpening (논문 best config) ────────────────
        #    한 번의 추가 high-pass step: Z += beta * (Z - Z @ G_s)
        if self.final_sharpening:
            Z = Z + self.idl_beta * (Z - Z @ G_s)

        self.Z = Z  # (U, I) — 최종 추천 점수 행렬

    # ------------------------------------------------------------------
    def forward(self, user_indices):
        return self.Z[user_indices]

    def calc_loss(self, batch_data):
        # Training-free 방법이므로 loss = 0
        return (torch.tensor(0.0, device=self.device),), None