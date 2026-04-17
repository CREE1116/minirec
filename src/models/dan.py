import torch
import numpy as np
from scipy import sparse
import gc
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
    
    res = float(num / den) if den > 0 else 0.0
    
    del G_np, degree, degree_sum, denom, sim, min_deg, w, mask
    gc.collect()
    
    return res


# ── EASE + DAN ────────────────────────────────────────────────────────────────

class EASE_DAN(BaseModel):
    """
    EASE with Data-Adaptive Normalization (DAN)
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_p             = config['model'].get('reg_p', 100.0)
        self.alpha_config      = config['model'].get('alpha', None)
        self.beta_config       = config['model'].get('beta',  None)
        self.volume_weight_exp = config['model'].get('volume_weight_exp', 1.5)
        self.max_items_hom     = config['model'].get('max_items_homophily', 5000)
        self.eps               = 1e-12
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting EASE_DAN (reg_p={self.reg_p}) on {self.device}...")
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr() # Hybrid inference

        item_counts = np.array(X_sp.sum(axis=0)).flatten().astype(np.float32)
        user_counts = np.array(X_sp.sum(axis=1)).flatten().astype(np.float32)

        gini = gini_coefficient(item_counts)
        alpha = self.alpha_config if self.alpha_config is not None else gini
        
        if self.beta_config is None:
            print("  computing edge homophily for beta (CPU)...")
            beta = edge_homophily(X_sp, self.volume_weight_exp, self.max_items_hom)
        else:
            beta = self.beta_config
            
        print(f"  Gini={gini:.4f}, α={alpha:.4f}, β={beta:.4f}")

        # 2. 유저 정규화 (X_tilde = D_U^{-β} X)
        print("  computing gram matrix with user normalization (CPU Sparse)...")
        # Optimization: use multiply directly instead of creating large D_U
        X_T_weighted = X_sp.multiply(np.power(user_counts + self.eps, -beta).reshape(-1, 1)).T
        G_np = X_T_weighted.dot(X_sp).toarray().astype(np.float32)
        
        del X_T_weighted
        gc.collect()

        if 'cuda' in str(self.device) and G_np.shape[0] < 20000:
            print("  inverting matrix (GPU)...")
            G_torch = torch.from_numpy(G_np).to(self.device)
            del G_np
            gc.collect()
            
            G_torch.diagonal().add_(self.reg_p)
            P_torch = torch.linalg.inv(G_torch)
            del G_torch
            
            diag_P = torch.diagonal(P_torch)
            W_gpu = -P_torch / (diag_P.reshape(1, -1) + self.eps)
            W_gpu.diagonal().zero_()
            del P_torch
            
            # 5. 아이템 정규화 (GPU)
            item_power = torch.from_numpy(np.power(item_counts + self.eps, -alpha)).to(self.device).float()
            self.weight_matrix = W_gpu * (1.0 / (item_power + self.eps)).reshape(-1, 1) * item_power.reshape(1, -1)
            self.weight_matrix.diagonal().zero_()
            del W_gpu, item_power
        else:
            print("  inverting matrix (CPU)...")
            G_np[np.diag_indices(self.n_items)] += self.reg_p
            P = np.linalg.inv(G_np)
            del G_np
            gc.collect()
            
            diag_P = np.diag(P)
            W = -P / (diag_P.reshape(1, -1) + self.eps)
            np.fill_diagonal(W, 0)
            del P
            
            # 5. 아이템 정규화 (CPU)
            item_power = np.power(item_counts + self.eps, -alpha)
            W = W * (1.0 / (item_power + self.eps)).reshape(-1, 1) * item_power.reshape(1, -1)
            np.fill_diagonal(W, 0)

            self.weight_matrix = torch.tensor(W, dtype=torch.float32, device=self.device)
            del W, item_power

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
        self.reg_p             = config['model'].get('reg_p', 100.0)
        self.dropout_p         = config['model'].get('dropout_p', 0.5)
        self.alpha_config      = config['model'].get('alpha', None)
        self.beta_config       = config['model'].get('beta',  None)
        self.volume_weight_exp = config['model'].get('volume_weight_exp', 1.5)
        self.max_items_hom     = config['model'].get('max_items_homophily', 5000)
        self.eps               = 1e-12
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting DLAE_DAN (reg_p={self.reg_p}, p={self.dropout_p}) on {self.device}...")
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr() # Hybrid inference

        item_counts = np.array(X_sp.sum(axis=0)).flatten().astype(np.float32)
        user_counts = np.array(X_sp.sum(axis=1)).flatten().astype(np.float32)

        gini = gini_coefficient(item_counts)
        alpha = self.alpha_config if self.alpha_config is not None else gini
        
        if self.beta_config is None:
            print("  computing edge homophily for beta (CPU)...")
            beta = edge_homophily(X_sp, self.volume_weight_exp, self.max_items_hom)
        else:
            beta = self.beta_config
            
        print(f"  Gini={gini:.4f}, α={alpha:.4f}, β={beta:.4f}")

        # 1. 유저 정규화
        print("  computing gram matrix with user normalization (CPU Sparse)...")
        X_T_weighted = X_sp.multiply(np.power(user_counts + self.eps, -beta).reshape(-1, 1)).T
        G_dan_np = X_T_weighted.dot(X_sp).toarray().astype(np.float32)
        
        del X_T_weighted
        gc.collect()

        # 2. DLAE Dropout 규제
        p_val = min(self.dropout_p, 0.99)
        lmbda_eff_np = self.reg_p + (p_val / (1.0 - p_val + self.eps)) * item_counts
        
        if 'cuda' in str(self.device) and G_dan_np.shape[0] < 20000:
            print("  solving linear system (GPU)...")
            G_rhs = torch.from_numpy(G_dan_np).to(self.device)
            G_lhs = G_rhs.clone()
            G_lhs.diagonal().add_(torch.from_numpy(lmbda_eff_np).to(self.device).float())
            
            W_gpu = torch.linalg.solve(G_lhs, G_rhs)
            del G_lhs, G_rhs, G_dan_np
            
            # 3. 아이템 정규화 후처리
            item_power = torch.from_numpy(np.power(item_counts + self.eps, -alpha)).to(self.device).float()
            self.weight_matrix = W_gpu * (1.0 / (item_power + self.eps)).reshape(-1, 1) * item_power.reshape(1, -1)
            self.weight_matrix.diagonal().zero_()
            del W_gpu, item_power
        else:
            print("  solving linear system (CPU)...")
            G_lhs = G_dan_np.copy()
            G_lhs[np.diag_indices(self.n_items)] += lmbda_eff_np
            
            W = np.linalg.solve(G_lhs, G_dan_np)
            del G_lhs, G_dan_np
            
            # 3. 아이템 정규화 후처리
            item_power = np.power(item_counts + self.eps, -alpha)
            W = W * (1.0 / (item_power + self.eps)).reshape(-1, 1) * item_power.reshape(1, -1)
            np.fill_diagonal(W, 0)

            self.weight_matrix = torch.tensor(W, dtype=torch.float32, device=self.device)
            del W, item_power

        gc.collect()
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        print("DLAE_DAN fitting complete.")

    def forward(self, user_indices):
        """Memory-efficient hybrid inference"""
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
