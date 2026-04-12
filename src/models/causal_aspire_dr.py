import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy

def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient [0, 1]. 높을수록 인기도 불균등."""
    values = np.asarray(values, dtype=float).flatten()
    if values.size == 0 or values.sum() == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = values.size
    cum = np.cumsum(sorted_vals)
    return (n + 1 - 2 * (cum.sum() / cum[-1])) / n

class CausalAspireDR(BaseModel):
    """
    Causal ASPIRE with Cross-Purification (CP) + DAN (Post-norm) + RLAE (Relaxation)
    - CPU-based solver for memory efficiency.
    - CP: Cross-purified weights for VST.
    - DAN: Gini-based item-side post-normalization.
    - RLAE: Relaxation parameter 'b' for diagonal constraints.
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        
        self.gamma      = config['model'].get('gamma', 0.5) 
        self.reg_lambda = config['model'].get('reg_lambda', 10.0) 
        self.b          = config['model'].get('b', 0.0) # RLAE relaxation: 0=EASE, 1=Ridge
        self.alpha_cfg  = config['model'].get('alpha', None) # DAN alpha (None = auto)
        
        self.eps        = 1e-12
        self.weight_matrix = None
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting Causal ASPIRE (CP + DAN + RLAE) on CPU...")
        print(f"  Params: gamma={self.gamma}, lambda={self.reg_lambda}, b={self.b}")

        X = get_train_matrix_scipy(data_loader)  # (U, K)
        X_sq = X.power(2)
        K = X.shape[1]

        # 1. Step 1: Cross-Purification (CP)
        print("  Step 1: Cross-Purification (CP)...")
        b_raw = np.asarray(X.sum(axis=0)).ravel()
        q_raw = np.asarray(X.sum(axis=1)).ravel()

        # [경로 A] 원본 아이템 편향으로 유저 정제
        D_I_raw_inv = sp.diags(np.power(b_raw + self.eps, -self.gamma))
        q_star = np.asarray(X_sq.dot(D_I_raw_inv).sum(axis=1)).ravel()

        # [경로 B] 원본 유저 편향으로 아이템 정제
        D_U_raw_inv = sp.diags(np.power(q_raw + self.eps, -self.gamma))
        b_star = np.asarray(X_sq.T.dot(D_U_raw_inv).sum(axis=1)).ravel()

        # 2. Step 2: VST Normalized Gram Matrix
        print("  Step 2: Computing Purified Gram Matrix (NumPy)...")
        user_scale = sp.diags(np.power(q_star + self.eps, -self.gamma / 2.0))
        item_scale = sp.diags(np.power(b_star + self.eps, -self.gamma / 2.0))

        # G_tilde = D_I_star @ (X.T @ D_U_star @ X) @ D_I_star
        # 메모리 효율을 위해 유저 스케일링을 먼저 적용 후 Gram 계산
        G_U = (X.T @ (user_scale @ X)).toarray()
        G_tilde = item_scale @ G_U @ item_scale

        # 3. Step 3: RLAE Solver (CPU)
        print(f"  Step 3: Solving RLAE (b={self.b}) on CPU...")
        A = G_tilde.copy()
        A[np.diag_indices(K)] += self.reg_lambda
        
        P = np.linalg.inv(A)
        diag_P = np.diag(P)
        
        # penalty = max(lambda, (1-b)/P_jj)
        penalty = np.maximum(self.reg_lambda, (1.0 - self.b) / (diag_P + self.eps))
        
        # W = I - P @ diag(penalty)
        W = - (P * penalty.reshape(1, -1))
        W[np.diag_indices(K)] += 1.0

        # 4. Step 4: DAN Post-normalization
        gini = gini_coefficient(b_raw)
        alpha = self.alpha_cfg if self.alpha_cfg is not None else gini
        print(f"  Step 4: DAN Post-norm (Gini={gini:.4f}, alpha={alpha:.4f})...")
        
        # W_final = D_I^{(1-alpha)} W D_I^{-(1-alpha)}
        item_power = np.power(b_raw + self.eps, -(1.0 - alpha))
        W = W * (1.0 / (item_power + self.eps)).reshape(-1, 1) * item_power.reshape(1, -1)
        np.fill_diagonal(W, 0.0)

        # 5. Move to Device
        self.weight_matrix = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_matrix_scipy = X
        print("Causal ASPIRE (CP+DAN+RLAE) fitting complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_tensor = torch.tensor(self.train_matrix_scipy[users].toarray(), 
                                    dtype=torch.float32, device=self.device)
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
