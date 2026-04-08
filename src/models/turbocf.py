import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel


class TurboCF(BaseModel):
    """
    Turbo-CF: Matrix Decomposition-Free Graph Filtering for Fast Recommendation
    SIGIR 2024 - Park et al.

    논문 핵심 알고리즘:
      1) Asymmetric normalization
             R̄ = D_U^{-α} R D_I^{-(1-α)}
      2) Item-item similarity graph (symmetric PSD)
             P̄ = R̄ᵀ R̄
      3) Row-normalize P̄ → P̂  (row-stochastic, eigenvalues ∈ [0, 1])
      4) Polynomial LPF H(P̂)  — matrix-decomposition 없이 polynomial로 근사
             LPF-1: H = P̂                       (linear)
             LPF-2: H = 2P̂ − P̂²               (2차 근사)
             LPF-3: H = 3P̂² − 2P̂³             (3차 smooth-step 근사, 논문 권장)
      5) 예측: Ŝ = R̄ @ H

    [기존 코드 vs 논문]
      - 기존: raw Gram + Tikhonov(λI) + geometric series (βᵏGᵏ)
        → 정규화 없음, 잘못된 필터 공식
      - 수정: asymmetric normalization → row-stochastic P̂ → polynomial H
        → 논문 Fig. 3 세 가지 LPF variant 구현

    하이퍼파라미터:
      alpha       : asymmetric normalization 강도 (default: 0.5)
                    0 → item-side only, 1 → user-side only
      filter_type : 1 / 2 / 3  (default: 3, 논문 권장)
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.alpha       = config['model'].get('alpha',       0.5)
        self.filter_type = config['model'].get('filter_type',   3)

    # ------------------------------------------------------------------
    def fit(self, data_loader):
        print(f"Fitting TurboCF  alpha={self.alpha}  filter_type={self.filter_type}")

        train_df         = data_loader.train_df
        n_users, n_items = data_loader.n_users, data_loader.n_items

        # ── Raw interaction matrix (sparse) ──────────────────────────
        R_sp = sp.csr_matrix(
            (np.ones(len(train_df)),
             (train_df['user_id'], train_df['item_id'])),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        # ── Asymmetric normalization: R̄ = D_U^{-α} R D_I^{-(1-α)} ──
        #    α=0.5이면 symmetric normalization (GF-CF와 동일)
        #    α ≠ 0.5이면 asymmetric (TurboCF 논문 구분점)
        rowsum = np.array(R_sp.sum(axis=1)).flatten() + 1e-12
        colsum = np.array(R_sp.sum(axis=0)).flatten() + 1e-12
        d_u    = np.power(rowsum, -self.alpha)           # (U,)
        d_i    = np.power(colsum, -(1.0 - self.alpha))   # (I,)
        R_bar_sp = sp.diags(d_u) @ R_sp @ sp.diags(d_i) # (U, I)

        # Dense로 변환 후 GPU 이동
        R_bar = torch.from_numpy(R_bar_sp.toarray()).float().to(self.device)  # (U, I)

        # ── Item-item similarity: P̄ = R̄ᵀ R̄  (symmetric PSD) ─────────
        P_bar = R_bar.t() @ R_bar                        # (I, I)

        # ── Row-normalize P̄ → P̂  (row-stochastic, eigenvalues ∈ [0,1])
        D_P   = P_bar.sum(dim=1, keepdim=True).clamp(min=1e-12)
        P_hat = P_bar / D_P                              # (I, I)

        # ── Polynomial LPF: H(P̂) ─────────────────────────────────────
        #
        #  모든 filter는 h(0)=0, h(1)=1 조건을 만족
        #  (eigenvalue=1 → 저주파 통과, eigenvalue=0 → 고주파 차단)
        #
        #  LPF-1: h(λ) = λ
        #  LPF-2: h(λ) = 2λ − λ²      (quadratic, h'(1)=0)
        #  LPF-3: h(λ) = 3λ² − 2λ³    (smooth-step, h'(0)=h'(1)=0)
        #          → 이상적 LPF(step function)에 가장 근접한 3차 다항식
        #
        if self.filter_type == 1:
            # LPF-1 (linear)
            H = P_hat

        elif self.filter_type == 2:
            # LPF-2 (second-order): H = 2P̂ − P̂²
            P2 = P_hat @ P_hat
            H  = 2.0 * P_hat - P2

        else:
            # LPF-3 (third-order smooth-step, 논문 권장): H = 3P̂² − 2P̂³
            P2 = P_hat @ P_hat
            P3 = P2   @ P_hat
            H  = 3.0 * P2 - 2.0 * P3

        self.H     = H      # (I, I) polynomial filter matrix
        self.R_bar = R_bar  # (U, I) normalized rating matrix (forward용)

    # ------------------------------------------------------------------
    def forward(self, user_indices):
        # Ŝ_u = r̄_u @ H  (논문: 정규화된 graph signal에 polynomial filter 적용)
        return self.R_bar[user_indices] @ self.H

    def calc_loss(self, batch_data):
        # Training-free 방법이므로 loss = 0
        return (torch.tensor(0.0, device=self.device),), None