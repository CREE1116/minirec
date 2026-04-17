import torch
import numpy as np
from scipy import sparse
import scipy.linalg as la
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

# ── DAN 유틸리티 ──────────────────────────────────────────────────────────────────

def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient [0, 1]. 높을수록 인기도 불균등."""
    values = np.asarray(values, dtype=np.float32).flatten()
    if values.size == 0 or values.sum() == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = values.size
    cum = np.cumsum(sorted_vals)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def edge_homophily(X: sparse.csr_matrix,
                   volume_weight_exp: float = 1.5,
                   max_items: int = 5000) -> float:
    """
    Edge homophily of item-item gram matrix (Efficient Block-wise version).
    """
    if X.shape[1] > max_items:
        rng = np.random.default_rng(42)
        idx = sorted(rng.choice(X.shape[1], size=max_items, replace=False))
        X = X[:, idx]

    n_items = X.shape[1]
    # Small items case: direct computation is OK
    if n_items <= 5000:
        G_np = X.T.dot(X).toarray().astype(np.float32)
    else:
        # Fallback to a simpler but safe path if somehow huge X is passed here
        return 0.5 

    degree = np.diag(G_np).copy().astype(np.float32)
    degree_sum = (degree[:, None] + degree[None, :]).astype(np.float32)

    denom = np.where(degree_sum - G_np == 0, np.float32(1e-12), degree_sum - G_np).astype(np.float32)
    sim = (G_np / denom).astype(np.float32)

    min_deg = np.where(
        np.minimum(degree[:, None], degree[None, :]) == 0,
        np.float32(1e-12),
        np.minimum(degree[:, None], degree[None, :])
    ).astype(np.float32)
    w = ((G_np ** np.float32(volume_weight_exp)) * (G_np / min_deg)).astype(np.float32)

    mask = np.triu(np.ones_like(G_np, dtype=bool), k=1) & (w > 0)
    num = (w * sim)[mask].sum()
    den = w[mask].sum()
    
    res = float(num / den) if den > 0 else 0.0
    
    del G_np, degree, degree_sum, denom, sim, min_deg, w, mask
    gc.collect()
    
    return res


# ── EASE + DAN ────────────────────────────────────────────────────────────────

class EASE_DAN(BaseModel):
    """
    EASE with Data-Adaptive Normalization (DAN)
    Optimized with Strict Minimal Memory.
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_p             = np.float32(config['model'].get('reg_p', 100.0))
        self.alpha_config      = config['model'].get('alpha', None)
        self.beta_config       = config['model'].get('beta',  None)
        self.volume_weight_exp = np.float32(config['model'].get('volume_weight_exp', 1.5))
        self.max_items_hom     = config['model'].get('max_items_homophily', 5000)
        self.eps               = np.float32(1e-12)
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting EASE_DAN on CPU with Strict minimal memory...")
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr() 

        item_counts = np.array(X_sp.sum(axis=0)).flatten().astype(np.float32)
        user_counts = np.array(X_sp.sum(axis=1)).flatten().astype(np.float32)

        gini = gini_coefficient(item_counts)
        alpha = np.float32(self.alpha_config if self.alpha_config is not None else gini)
        
        if self.beta_config is None:
            print("  computing edge homophily for beta...")
            beta = np.float32(edge_homophily(X_sp, self.volume_weight_exp, self.max_items_hom))
        else:
            beta = np.float32(self.beta_config)
            
        print(f"  Gini={gini:.4f}, α={alpha:.4f}, β={beta:.4f}")

        # 2. 유저 정규화 (X_tilde = D_U^{-β} X)
        print("  computing gram matrix (CPU Block-wise float32)...")
        u_weights = np.power(user_counts + self.eps, -beta).astype(np.float32)
        
        # Use optimized block-wise constructor
        G_np = compute_gram_matrix(X_sp, data_loader, weights=u_weights)
        
        del u_weights, user_counts
        gc.collect()

        print("  inverting matrix (CPU In-place NumPy float32)...")
        G_np[np.diag_indices(self.n_items)] += self.reg_p
        P = la.inv(G_np, overwrite_a=True).astype(np.float32)
        del G_np
        gc.collect()
        
        diag_P = np.diag(P).astype(np.float32)
        W = (-P / (diag_P.reshape(1, -1) + self.eps)).astype(np.float32)
        np.fill_diagonal(W, 0)
        del P, diag_P
        gc.collect()
        
        # 5. 아이템 정규화 (CPU)
        item_power = np.power(item_counts + self.eps, -alpha).astype(np.float32)
        W = (W * (np.float32(1.0) / (item_power + self.eps)).reshape(-1, 1) * item_power.reshape(1, -1)).astype(np.float32)
        np.fill_diagonal(W, 0)

        self.weight_matrix = torch.tensor(W, dtype=torch.float32, device=self.device)
        del W, item_power, item_counts
        gc.collect()

        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        print("EASE_DAN fitting complete.")

    def forward(self, user_indices):
        """Memory-efficient hybrid inference"""
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None


# ── DLAE + DAN ────────────────────────────────────────────────────────────────

class DLAE_DAN(BaseModel):
    """
    DLAE with Data-Adaptive Normalization (DAN)
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_p             = np.float32(config['model'].get('reg_p', 100.0))
        self.dropout_p         = np.float32(config['model'].get('dropout_p', 0.5))
        self.alpha_config      = config['model'].get('alpha', None)
        self.beta_config       = config['model'].get('beta',  None)
        self.volume_weight_exp = np.float32(config['model'].get('volume_weight_exp', 1.5))
        self.max_items_hom     = config['model'].get('max_items_homophily', 5000)
        self.eps               = np.float32(1e-12)
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting DLAE_DAN on CPU with Strict minimal memory...")
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr()

        item_counts = np.array(X_sp.sum(axis=0)).flatten().astype(np.float32)
        user_counts = np.array(X_sp.sum(axis=1)).flatten().astype(np.float32)

        gini = gini_coefficient(item_counts)
        alpha = np.float32(self.alpha_config if self.alpha_config is not None else gini)
        
        if self.beta_config is None:
            print("  computing edge homophily for beta...")
            beta = np.float32(edge_homophily(X_sp, self.volume_weight_exp, self.max_items_hom))
        else:
            beta = self.beta_config
            
        print(f"  Gini={gini:.4f}, α={alpha:.4f}, β={beta:.4f}")

        # 1. 유저 정규화
        print("  computing gram matrix (CPU Block-wise float32)...")
        u_weights = np.power(user_counts + self.eps, -beta).astype(np.float32)
        G_dan_np = compute_gram_matrix(X_sp, data_loader, weights=u_weights)
        
        del u_weights, user_counts
        gc.collect()

        # 2. DLAE Dropout 규제
        p_val = self.dropout_p
        lmbda_eff_np = (self.reg_p + (p_val / (np.float32(1.0) - p_val + self.eps)) * item_counts).astype(np.float32)
        
        print("  solving linear system (CPU In-place NumPy float32)...")
        G_lhs = G_dan_np.astype(np.float32)
        G_lhs[np.diag_indices(self.n_items)] += lmbda_eff_np
        
        # Original Gram for RHS
        G_rhs = compute_gram_matrix(X_sp, data_loader, weights=np.power(np.array(X_sp.sum(axis=1)).flatten() + self.eps, -beta).astype(np.float32))

        W = la.solve(G_lhs, G_rhs, overwrite_a=True, overwrite_b=True).astype(np.float32)
        del G_lhs, G_rhs, lmbda_eff_np
        gc.collect()
        
        # 3. 아이템 정규화 후처리
        item_power = np.power(item_counts + self.eps, -alpha).astype(np.float32)
        W = (W * (np.float32(1.0) / (item_power + self.eps)).reshape(-1, 1) * item_power.reshape(1, -1)).astype(np.float32)
        np.fill_diagonal(W, 0)

        self.weight_matrix = torch.tensor(W, dtype=torch.float32, device=self.device)
        del W, item_power, item_counts
        gc.collect()

        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        print("DLAE_DAN fitting complete.")

    def forward(self, user_indices):
        """Memory-efficient hybrid inference"""
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
