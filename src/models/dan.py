import torch
import numpy as np
from .base import BaseModel


# ── DAN 모듈 ──────────────────────────────────────────────────────────────────

def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient [0, 1]. 높을수록 인기도 불균등."""
    values = np.asarray(values, dtype=float).flatten()
    if values.size == 0 or values.sum() == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = values.size
    cum = np.cumsum(sorted_vals)
    return (n + 1 - 2 * (cum.sum() / cum[-1])) / n


def edge_homophily(G_np: np.ndarray,
                   volume_weight_exp: float = 1.5,
                   max_items: int = 5000) -> float:
    """
    Edge homophily of item-item gram matrix.

    h = Σ_{i<j} w_ij * sim_ij / Σ_{i<j} w_ij

    sim_ij = G_ij / (d_i + d_j - G_ij)   (Jaccard-like)
    w_ij   = G_ij^v * (G_ij / min(d_i, d_j))

    K > max_items이면 서브샘플링 (메모리 절약)
    """
    K = G_np.shape[0]
    if K > max_items:
        rng = np.random.default_rng(42)
        idx = rng.choice(K, size=max_items, replace=False)
        G_np = G_np[np.ix_(idx, idx)]

    degree = np.diag(G_np).copy()
    degree_sum = degree[:, None] + degree[None, :]

    denom = np.where(degree_sum - G_np == 0, 1e-12, degree_sum - G_np)
    sim = G_np / denom

    min_deg = np.where(
        np.minimum(degree[:, None], degree[None, :]) == 0,
        1e-12,
        np.minimum(degree[:, None], degree[None, :])
    )
    w = (G_np ** volume_weight_exp) * (G_np / min_deg)

    mask = np.triu(np.ones_like(G_np, dtype=bool), k=1) & (w > 0)
    num = (w * sim)[mask].sum()
    den = w[mask].sum()
    return float(num / den) if den > 0 else 0.0


def apply_dan(G: torch.Tensor,
              X: torch.Tensor,
              alpha=None,
              beta=None,
              volume_weight_exp: float = 1.5,
              max_items_homophily: int = 5000,
              eps: float = 1e-12):
    """
    DAN 핵심 연산.

    Returns
    -------
    G_dan   : 유저 정규화된 gram matrix
    info    : {'alpha', 'beta', 'gini', 'homophily'}
    n_i     : item counts (numpy)
    alpha   : 최종 사용된 alpha
    """
    device = G.device
    G_np = G.detach().cpu().numpy()
    X_dense = (X.to_dense() if X.is_sparse else X).detach().cpu().numpy()

    n_i = G_np.diagonal().copy()
    n_u = X_dense.sum(axis=1)

    # α: Gini of item counts
    gini = gini_coefficient(n_i)
    alpha_used = gini if alpha is None else alpha

    # β: edge homophily
    hom = edge_homophily(G_np, volume_weight_exp, max_items_homophily)
    beta_used = hom if beta is None else beta

    info = {'alpha': alpha_used, 'beta': beta_used,
            'gini': gini, 'homophily': hom}

    # 유저 정규화: X_tilde = D_U^{-β} X
    u_weight = np.power(n_u + eps, -beta_used)
    X_tilde  = X_dense * u_weight[:, None]

    G_dan_np = X_tilde.T @ X_tilde
    G_dan = torch.tensor(G_dan_np, dtype=torch.float32, device=device)

    return G_dan, info, n_i, alpha_used


def apply_item_norm(W: torch.Tensor,
                    n_i: np.ndarray,
                    alpha: float,
                    eps: float = 1e-12) -> torch.Tensor:
    """
    DAN 아이템 정규화 후처리.

    W_final[i,j] = W[i,j] * (1/n_i^{-(1-α)}) * n_j^{-(1-α)}
                 = W[i,j] * n_i^{(1-α)} * n_j^{-(1-α)}
    """
    device = W.device
    item_power = torch.tensor(
        np.power(n_i + eps, -(1 - alpha)),
        dtype=torch.float32, device=device
    )
    return W * (1.0 / (item_power + eps)).unsqueeze(1) * item_power.unsqueeze(0)


# ── EASE + DAN ────────────────────────────────────────────────────────────────

class EASE_DAN(BaseModel):
    """
    EASE with Data-Adaptive Normalization (DAN)
    Park et al., SIGIR 2025.

    α = Gini(n_i)         → 아이템 인기도 불균등도 (자동)
    β = EdgeHomophily(G)  → 이웃 유사도 (자동)

    G_dan  = (D_U^{-β} X)^T (D_U^{-β} X)
    W_ease = EASE(G_dan)
    W      = D_I^{(1-α)} W_ease D_I^{-(1-α)}
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda        = config['model'].get('reg_lambda', 100.0)
        self.alpha_config      = config['model'].get('alpha', None)
        self.beta_config       = config['model'].get('beta',  None)
        self.volume_weight_exp = config['model'].get('volume_weight_exp', 1.5)
        self.max_items_hom     = config['model'].get('max_items_homophily', 5000)
        self.eps               = 1e-12
        self.weight_matrix = None
        self.train_matrix  = None

    def fit(self, data_loader):
        print(f"Fitting EASE_DAN (lambda={self.reg_lambda}) on {self.device}...")
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X

        G = torch.sparse.mm(X.t(), X.to_dense()).to(self.device)

        G_dan, info, n_i, alpha = apply_dan(
            G, X,
            alpha=self.alpha_config, beta=self.beta_config,
            volume_weight_exp=self.volume_weight_exp,
            max_items_homophily=self.max_items_hom,
        )
        print(f"  Gini={info['gini']:.4f}, Homophily={info['homophily']:.4f}")
        print(f"  α={info['alpha']:.4f}, β={info['beta']:.4f}")

        A = G_dan + self.reg_lambda * torch.eye(G.shape[0], device=self.device)
        try:
            P = torch.linalg.inv(A)
        except (torch._C._LinAlgError, RuntimeError):
            A.diagonal().add_(self.reg_lambda * 10 + 1e-4)
            P = torch.linalg.inv(A)

        diag_P = P.diagonal()
        W = P / (-diag_P + self.eps)
        W.diagonal().zero_()
        W = apply_item_norm(W, n_i, alpha, self.eps)
        W.diagonal().zero_()

        self.weight_matrix = W
        print("EASE_DAN fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)
        return self.train_matrix_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None


# ── DLAE + DAN ────────────────────────────────────────────────────────────────

class DLAE_DAN(BaseModel):
    """
    DLAE with Data-Adaptive Normalization (DAN)

    EASE_DAN + dropout 앙상블 규제 (DLAE 스타일)
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda        = config['model'].get('reg_lambda', 100.0)
        self.dropout_p         = config['model'].get('dropout_p', 0.5)
        self.alpha_config      = config['model'].get('alpha', None)
        self.beta_config       = config['model'].get('beta',  None)
        self.volume_weight_exp = config['model'].get('volume_weight_exp', 1.5)
        self.max_items_hom     = config['model'].get('max_items_homophily', 5000)
        self.eps               = 1e-12
        self.weight_matrix = None
        self.train_matrix  = None

    def fit(self, data_loader):
        print(f"Fitting DLAE_DAN (lambda={self.reg_lambda}, "
              f"dropout={self.dropout_p}) on {self.device}...")
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X

        G = torch.sparse.mm(X.t(), X.to_dense()).to(self.device)

        G_dan, info, n_i, alpha = apply_dan(
            G, X,
            alpha=self.alpha_config, beta=self.beta_config,
            volume_weight_exp=self.volume_weight_exp,
            max_items_homophily=self.max_items_hom,
        )
        print(f"  Gini={info['gini']:.4f}, Homophily={info['homophily']:.4f}")
        print(f"  α={info['alpha']:.4f}, β={info['beta']:.4f}")

        # DLAE 규제
        n_i_t = torch.tensor(n_i, dtype=torch.float32, device=self.device)
        p = min(self.dropout_p, 0.99)
        w = (p / (1.0 - p)) * n_i_t

        A = G_dan.clone()
        A.diagonal().add_(w + self.reg_lambda)
        try:
            P = torch.linalg.inv(A)
        except (torch._C._LinAlgError, RuntimeError):
            A.diagonal().add_(self.reg_lambda * 10 + 1e-4)
            P = torch.linalg.inv(A)

        lmbda = w + self.reg_lambda
        W = torch.eye(G.shape[0], device=self.device) - P * lmbda.unsqueeze(0)
        W.diagonal().zero_()
        W = apply_item_norm(W, n_i, alpha, self.eps)
        W.diagonal().zero_()

        self.weight_matrix = W
        print("DLAE_DAN fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)
        return self.train_matrix_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
