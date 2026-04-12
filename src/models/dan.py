import torch
import numpy as np
from scipy import sparse
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy

# ── DAN 유틸리티 ──────────────────────────────────────────────────────────────────

def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient [0, 1]. 높을수록 인기도 불균등."""
    values = np.asarray(values, dtype=float).flatten()
    if values.size == 0 or values.sum() == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = values.size
    cum = np.cumsum(sorted_vals)
    return (n + 1 - 2 * (cum.sum() / cum[-1])) / n


def edge_homophily(X: sparse.csr_matrix,
                   volume_weight_exp: float = 1.5,
                   max_items: int = 5000) -> float:
    """
    Edge homophily of item-item gram matrix (Efficient Scipy version).
    """
    if X.shape[1] > max_items:
        rng = np.random.default_rng(42)
        idx = sorted(rng.choice(X.shape[1], size=max_items, replace=False))
        X = X[:, idx]

    # Gram matrix calculation on CPU
    G_np = X.T.dot(X).toarray().astype(np.float32)
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


# ── EASE + DAN ────────────────────────────────────────────────────────────────

class EASE_DAN(BaseModel):
    """
    EASE with Data-Adaptive Normalization (DAN)
    CPU-based solver for stability and VRAM efficiency.
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
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting EASE_DAN (lambda={self.reg_lambda}) on CPU...")
        X = get_train_matrix_scipy(data_loader)
        self.train_matrix_scipy = X

        item_counts = np.array(X.sum(axis=0)).flatten().astype(np.float32)
        user_counts = np.array(X.sum(axis=1)).flatten().astype(np.float32)

        # 1. 자동 alpha (Gini), beta (Homophily) 계산
        gini = gini_coefficient(item_counts)
        alpha = self.alpha_config if self.alpha_config is not None else gini
        
        if self.beta_config is None:
            print("  computing edge homophily for beta (CPU)...")
            beta = edge_homophily(X, self.volume_weight_exp, self.max_items_hom)
        else:
            beta = self.beta_config
            
        print(f"  Gini={gini:.4f}, α={alpha:.4f}, β={beta:.4f}")

        # 2. 유저 정규화 (X_tilde = D_U^{-β} X) 및 Gram 계산
        print("  computing gram matrix with user normalization (CPU)...")
        # Sparse multiply is efficient
        X_T_weighted = X.multiply(np.power(user_counts + self.eps, -beta).reshape(-1, 1)).T
        G_dan = X_T_weighted.dot(X).toarray().astype(np.float32)

        # 3. CPU Solver (NumPy)
        A = G_dan.copy()
        A[np.diag_indices(self.n_items)] += self.reg_lambda
        
        print("  inverting matrix on CPU...")
        P = np.linalg.inv(A)
        diag_P = np.diag(P)
        
        # W = -P / P_jj
        W = P / (-diag_P.reshape(-1, 1) + self.eps)
        np.fill_diagonal(W, 0)
        
        # 4. 아이템 정규화 후처리
        # W_final = D_I^{(1-α)} W D_I^{-(1-α)}
        item_power = np.power(item_counts + self.eps, -(1.0 - alpha))
        W = W * (1.0 / (item_power + self.eps)).reshape(-1, 1) * item_power.reshape(1, -1)
        np.fill_diagonal(W, 0)

        # 5. Move results to Device for Forward
        self.weight_matrix = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_matrix_gpu = self.get_train_matrix(data_loader)
        print("EASE_DAN fitting complete.")

    def forward(self, user_indices):
        input_tensor = torch.index_select(self.train_matrix_gpu, 0, user_indices).to_dense()
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None


# ── DLAE + DAN ────────────────────────────────────────────────────────────────

class DLAE_DAN(BaseModel):
    """
    DLAE with Data-Adaptive Normalization (DAN)
    CPU-based solver for stability and VRAM efficiency.
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
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting DLAE_DAN (lambda={self.reg_lambda}, p={self.dropout_p}) on CPU...")
        X = get_train_matrix_scipy(data_loader)
        self.train_matrix_scipy = X

        item_counts = np.array(X.sum(axis=0)).flatten().astype(np.float32)
        user_counts = np.array(X.sum(axis=1)).flatten().astype(np.float32)

        gini = gini_coefficient(item_counts)
        alpha = self.alpha_config if self.alpha_config is not None else gini
        
        if self.beta_config is None:
            print("  computing edge homophily for beta (CPU)...")
            beta = edge_homophily(X, self.volume_weight_exp, self.max_items_hom)
        else:
            beta = self.beta_config
            
        print(f"  Gini={gini:.4f}, α={alpha:.4f}, β={beta:.4f}")

        # 1. 유저 정규화
        print("  computing gram matrix with user normalization (CPU)...")
        X_T_weighted = X.multiply(np.power(user_counts + self.eps, -beta).reshape(-1, 1)).T
        G_dan = X_T_weighted.dot(X).toarray().astype(np.float32)

        # 2. DLAE Dropout 규제 (CPU Solver)
        # lmbda_eff = reg_lambda + (p/(1-p)) * item_counts
        p_val = min(self.dropout_p, 0.99)
        w_dropout = (p_val / (1.0 - p_val)) * item_counts
        
        A = G_dan.copy()
        A[np.diag_indices(self.n_items)] += (w_dropout + self.reg_lambda)
        
        print("  solving linear system on CPU...")
        # W = (G_dan + diag(w_dropout + lambda))^{-1} G_dan
        W = np.linalg.solve(A, G_dan)
        
        # 3. 아이템 정규화 후처리
        item_power = np.power(item_counts + self.eps, -(1.0 - alpha))
        W = W * (1.0 / (item_power + self.eps)).reshape(-1, 1) * item_power.reshape(1, -1)
        np.fill_diagonal(W, 0)

        # 4. Move to Device
        self.weight_matrix = torch.tensor(W, dtype=torch.float32, device=self.device)
        print("DLAE_DAN fitting complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_matrix = self.train_matrix_scipy[users].toarray()
        input_tensor = torch.tensor(input_matrix, dtype=torch.float32, device=self.device)
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
