import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel
from src.utils.svd import get_svd_cache


class GF_CF(BaseModel):
    """
    How Powerful is Graph Convolution for Recommendation? (GF-CF)
    CIKM 2021 - Shen et al.

    논문 핵심 수식 (Eq. 8):
        s_u = r̃_u  (G + α · V_k Vₖᵀ)

    구성:
      · R̃ = D_U^{-0.5} R D_I^{-0.5}   (symmetric normalization)
      · G  = R̃ᵀ R̃                      (linear LPF, item × item)
      · V_k = top-k right singular vectors of R̃  (ideal LPF)
      · W = G + α · V_k Vₖᵀ            (combined weight matrix)
      · 예측: r̃_u @ W  (정규화된 user 행을 graph signal로 사용)

    [기존 코드 vs 논문]
      - 기존: forward에서 raw R 사용  →  수정: 정규화된 R̃ 사용
        (논문은 r̃_u를 graph signal로 정의)
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.k     = config['model'].get('k',     256)
        self.alpha = config['model'].get('alpha', 0.3)

    # ------------------------------------------------------------------
    def fit(self, data_loader):
        print(f"Fitting GF-CF  k={self.k}  alpha={self.alpha}")

        train_df       = data_loader.train_df
        n_users, n_items = data_loader.n_users, data_loader.n_items

        # ── Raw interaction matrix (sparse) ──────────────────────────
        R_sp = sp.csr_matrix(
            (np.ones(len(train_df)),
             (train_df['user_id'], train_df['item_id'])),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        # ── Symmetric normalization: R̃ = D_U^{-0.5} R D_I^{-0.5} ───
        rowsum = np.array(R_sp.sum(axis=1)).flatten()
        colsum = np.array(R_sp.sum(axis=0)).flatten()
        d_u    = np.power(rowsum + 1e-12, -0.5)   # (U,)
        d_i    = np.power(colsum + 1e-12, -0.5)   # (I,)
        R_tilde_sp = sp.diags(d_u) @ R_sp @ sp.diags(d_i)  # (U, I)

        # ── Linear LPF : G = R̃ᵀ R̃  (item × item) ───────────────────
        R_tilde_torch = self._to_torch_sparse(R_tilde_sp).to(self.device)
        G = torch.sparse.mm(                          # (I, I)
            R_tilde_torch.t(),
            R_tilde_torch.to_dense(),
        )

        # ── Ideal LPF : V_k Vₖᵀ  (top-k right singular vectors of R̃)
        #    논문: V_k ∈ R^{I × k},  ideal LPF matrix = V_k Vₖᵀ  (I × I)
        svd_res  = get_svd_cache(
            data_loader, k_max=self.k,
            matrix=R_tilde_sp, cache_id="normalized",
        )
        V        = torch.from_numpy(svd_res['vt'].T).to(self.device)  # (I, k)
        S_global = V @ V.t()                                          # (I, I)

        # ── Combined weight matrix: W = G + α · S_global ─────────────
        self.weight_matrix = G + self.alpha * S_global                # (I, I)

        # ── 정규화된 R̃ 저장 (forward에서 graph signal로 사용) ─────────
        #    논문: s_u = r̃_u @ W  (raw r_u 가 아님)
        self.R_tilde_dense = torch.from_numpy(
            R_tilde_sp.toarray()
        ).to(self.device)                                              # (U, I)

    # ------------------------------------------------------------------
    def _to_torch_sparse(self, sp_mat):
        sp_mat  = sp_mat.tocoo()
        indices = torch.from_numpy(
            np.vstack((sp_mat.row, sp_mat.col)).astype(np.int64)
        )
        values = torch.from_numpy(sp_mat.data)
        shape  = torch.Size(sp_mat.shape)
        return torch.sparse_coo_tensor(indices, values, shape).coalesce()

    # ------------------------------------------------------------------
    def forward(self, user_indices):
        # s_u = r̃_u @ W   (논문 Eq. 8)
        return self.R_tilde_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None