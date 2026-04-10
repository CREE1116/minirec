import torch
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix
import torch
import numpy as np

import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy

class AspirePure(BaseModel):
    """
    Dual-ASPIRE: Dual-Sided Topological Normalization CF
    - User-side: IPW Normalization (1/n_u) via sparse matrix dot.
    - Item-side: Topological Entropy Normalization (Shannon Entropy).
    - Unconstrained Inversion: Preserves long-tail without 1/P_ii penalty.
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        # alpha: 엔트로피 페널티 강도 조절 (이론상 1.0이 순수 IPW 폼)
        self.alpha      = config['model'].get('alpha', 1.0) 
        self.eps        = 1e-12
        self.weight_matrix = None
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting Pure Causal ASPIRE (lambda={self.reg_lambda}, alpha={self.alpha}) on {self.device}...")
        
        X_np = get_train_matrix_scipy(data_loader).toarray()
        self.train_matrix = X_np 
        X = torch.tensor(X_np, dtype=torch.float32, device=self.device)

        # ── 1. User-side IPW Normalization (G^U) ────────────────────────────
        n_u = X.sum(dim=1, keepdim=True) + self.eps            
        D_U_inv = 1.0 / n_u                                    
        
        # G^U_ij = sum_u (X_ui * X_uj) / n_u
        G_U = X.T @ (X * D_U_inv)                              

        # ── 2. Item-side Popularity Proxy (A_i) ─────────────────────────────
        # G^U의 대각 성분이 곧 유저 편향이 통제된 진성 인기도(A_i)임
        A_i = G_U.diagonal()
        
        # D_A^{-alpha/2} 계산
        D_A_inv_sqrt = torch.pow(A_i + self.eps, -self.alpha / 2.0)

        # ── 3. Dual-Sided Symmetrical Normalization ─────────────────────────
        # G_tilde = D_A^{-alpha/2} * G^U * D_A^{-alpha/2}
        G_tilde = G_U * D_A_inv_sqrt.view(-1, 1) * D_A_inv_sqrt.view(1, -1)

        # Trace 보존 (람다 스케일 유지용)
        trace_ratio = G_U.trace() / (G_tilde.trace() + self.eps)
        G_tilde = G_tilde * trace_ratio

        # ── 4. Unconstrained Ridge Regression ───────────────────────────────
        A = G_tilde.clone()
        A.diagonal().add_(self.reg_lambda)                     
        
        P_inv = torch.linalg.inv(A)                            
        
        W = -P_inv                                             
        W.diagonal().zero_()                                   
        
        self.weight_matrix = W
        print("Pure Causal ASPIRE fitting complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_tensor = torch.tensor(self.train_matrix[users], dtype=torch.float32, device=self.device)
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None