import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy
from time import time

class CausalAspireDR(BaseModel):
    """
    Causal ASPIRE + 2-Stage Doubly Robust (Causal-DR)
    - Stage 1: Compute Causal Space (G_tilde) and Base Estimator (W_1)
    - Stage 2: Compute Asymmetric IPW Matrix (G_IPW)
    - Stage 3: Closed-form Convex Interpolation for Doubly Robustness
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        
        # ──────────────────────────────────────────────────────────
        # 🔥 인과적 대통일 상수 gamma (기존 alpha, beta를 하나로 통합)
        # ──────────────────────────────────────────────────────────
        self.gamma      = config['model'].get('gamma', 0.5) 
        # config에 gamma가 없고 alpha만 있다면 하위 호환성 유지
            
        self.reg_lambda = config['model'].get('reg_lambda', 10.0) 
        # self.num_iters  = config['model'].get('num_iters', 10)
        self.eps        = 1e-12
        self.beta = 1.5
        self.weight_matrix = None
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting Causal ASPIRE (DR + Symmetric Normalization)...")

        X = get_train_matrix_scipy(data_loader)  # (U, K)
        K = X.shape[1]

        # 1. 기존 Global Scaling (raw degree symmetric) ── 유지
        q_u = np.asarray(X.sum(axis=1)).ravel()
        b_i = np.asarray(X.sum(axis=0)).ravel()

        u_weight = np.power(q_u + self.eps, -self.gamma)
        i_weight = np.power(b_i + self.eps, -self.gamma / 2.0)

        D_U_inv = sp.diags(u_weight)
        D_I_inv_half = sp.diags(i_weight)

        G_ips_raw = (X.T @ D_U_inv @ X).toarray()          # Raw IPS Gram

        # 2. Direct Model (simple bias imputation) ── DR의 핵심 킥
        # (user bias + item bias + global mean)
        global_mean = X.mean()
        user_bias = np.asarray(X.mean(axis=1)).ravel() - global_mean
        item_bias = np.asarray(X.mean(axis=0)).ravel() - global_mean

        # imputation matrix (sparse → dense)
        # R_hat_ui = user_bias_u + item_bias_i + global_mean
        R_hat = np.outer(user_bias, np.ones(K)) + \
                np.outer(np.ones(X.shape[0]), item_bias) + global_mean
        R_hat = np.clip(R_hat, 0, 1)   # implicit feedback이니 0~1로

        # 3. DR Gram (IPS + correction)
        # G_dr = G_ips + sum_u w_u (R_hat_u.T @ R_hat_u - (R_hat_u * O_u).T @ (R_hat_u * O_u))
        obs_mask = (X.toarray() > 0).astype(float)
        
        # Weighted model Gram: sum_u w_u R_hat_u.T @ R_hat_u
        # D_U_inv @ R_hat scales each user's imputation by its weight w_u
        G_model_raw = R_hat.T @ (D_U_inv @ R_hat)
        
        # Weighted imputed-observed Gram: sum_u w_u (R_hat_u * O_u).T @ (R_hat_u * O_u)
        R_hat_obs = R_hat * obs_mask
        G_imputed_obs_raw = R_hat_obs.T @ (D_U_inv @ R_hat_obs)
        
        G_dr_raw = G_ips_raw + G_model_raw - G_imputed_obs_raw
        
        # symmetric normalization (item-side)
        G_dr = D_I_inv_half @ G_dr_raw @ D_I_inv_half

        # (선택) SNIPS-style trace normalization으로 variance 추가 안정화
        trace_norm = np.trace(G_dr)
        if trace_norm > 1e-8:
            G_dr /= trace_norm

        # 4. EASE Solver (기존과 동일)
        G_torch = torch.tensor(G_dr, dtype=torch.float32, device=self.device)
        A_mat = G_torch + self.reg_lambda * torch.eye(K, device=self.device)
        P = torch.linalg.inv(A_mat)
        P_diag = P.diagonal()
        W = -P / (P_diag.unsqueeze(0) + self.eps)
        W.fill_diagonal_(0.0)

        self.weight_matrix = W
        self.train_matrix_scipy = X
        print("DR Hybrid Causal ASPIRE complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_tensor = torch.tensor(self.train_matrix_scipy[users].toarray(), 
                                    dtype=torch.float32, device=self.device)
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None