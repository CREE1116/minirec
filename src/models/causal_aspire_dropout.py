import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy

class CausalAspireDropout(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 10.0)
        self.alpha      = config['model'].get('alpha', 0.5)
        self.beta       = config['model'].get('beta', 0.5)
        self.dropout_p  = config['model'].get('dropout_p', 0.3)  # 추가
        self.eps        = 1e-12

        self.weight_matrix = None
        self.train_matrix_scipy = None

    def fit(self, data_loader):
        print(f"Fitting Causal ASPIRE + Dropout (λ={self.reg_lambda}, α={self.alpha}, β={self.beta}, p={self.dropout_p})")

        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_scipy = X_sp

        n_u = np.asarray(X_sp.sum(axis=1)).ravel()
        K   = X_sp.shape[1]

        # ── Stage 1: User IPW ──
        user_weights = np.power(n_u + self.eps, -self.beta)
        D_U_inv = sp.diags(user_weights)

        X_weighted = D_U_inv @ X_sp
        G_U = (X_sp.T @ X_weighted).toarray()

        # ── Stage 2: Item normalization ──
        A_i = G_U.diagonal().copy()
        scale = np.power(A_i + self.eps, -self.alpha / 2.0)
        G_tilde = G_U * scale[:, None] * scale[None, :]

        # ── 🔥 Dropout Regularization ──
        p = min(self.dropout_p, 0.99)
        w_dropout = (p / (1.0 - p)) * A_i  # 핵심

        # ── Stage 3: Solve ──
        G_torch = torch.tensor(G_tilde, dtype=torch.float32, device=self.device)

        # diagonal에 dropout 추가
        A_mat = G_torch + torch.diag(
            torch.tensor(w_dropout + self.reg_lambda, dtype=torch.float32, device=self.device)
        )

        try:
            W = torch.linalg.solve(A_mat, G_torch)
        except RuntimeError:
            print("[Warning] fallback regularization")
            A_mat.diagonal().add_(self.reg_lambda * 10 + 1e-4)
            W = torch.linalg.solve(A_mat, G_torch)

        self.weight_matrix = W
        print("Causal ASPIRE + Dropout fitting complete.")

    def forward(self, user_indices):
        users = user_indices.cpu().numpy()
        input_tensor = torch.tensor(self.train_matrix_scipy[users].toarray(), dtype=torch.float32, device=self.device)
        return input_tensor @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None