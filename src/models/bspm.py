import torch
from .base import BaseModel


class BSPM(BaseModel):
    """
    BSPM (improved, practical version)

    Key fixes:
    - Avoid full inverse during blurring
    - Use iterative solver for blurring
    - Reduce dense usage where possible
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)

        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.t = config['model'].get('t', 1.0)
        self.s = config['model'].get('s', 0.5)

        self.blur_iter = config['model'].get('blur_iter', 10)

        self.eps = 1e-12

        self.weight_matrix = None
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting BSPM (lambda={self.reg_lambda}, t={self.t}, s={self.s}) on {self.device}...")

        X = self.get_train_matrix(data_loader)
        self.train_matrix = X

        # BSPM requires dense matrix for item-item relationships
        X_dense = X.to_dense().to(self.device)

        # ── 1. Degree calculation ─────────────────────────────────────────
        d_u = X_dense.sum(dim=1)
        d_i = X_dense.sum(dim=0)

        inv_sqrt_u = torch.pow(d_u + self.eps, -0.5)
        inv_sqrt_i = torch.pow(d_i + self.eps, -0.5)

        # ── 2. Normalized interaction ─────────────────────────────────────
        X_norm = X_dense * inv_sqrt_u.unsqueeze(1) * inv_sqrt_i.unsqueeze(0)

        # ── 3. Normalized Gram ────────────────────────────────────────────
        G = X_norm.t() @ X_norm   # I x I

        # ── 4. Blurring (iterative solve) ─────────────────────────────────
        # Approximate the solution to ((1+t)I - tG) Z = X_norm
        # using a Jacobi-like iterative method to avoid full matrix inversion.
        Z = X_norm.clone()

        for _ in range(self.blur_iter):
            Z = X_norm + self.t * (Z @ G) / (1.0 + self.t)

        X_blur = Z

        # ── 5. Sharpening ──────────────────────────────────────────────────
        # X_sharp = X_norm + s * (X_norm - X_blur)
        X_sharp = (1.0 + self.s) * X_norm - self.s * X_blur

        # ── 6. Gram (sharpened) ────────────────────────────────────────────
        G_sharp = X_sharp.t() @ X_sharp

        # ── 7. EASE Solution ───────────────────────────────────────────────
        A = G_sharp.clone()
        A.diagonal().add_(self.reg_lambda)

        try:
            P = torch.linalg.inv(A)
        except RuntimeError:
            print("[Warning] Singular matrix, applying fallback regularization.")
            A.diagonal().add_(self.reg_lambda * 10 + 1e-4)
            P = torch.linalg.inv(A)

        diag_P = P.diagonal()

        # W = I - P / diag(P)
        W = P / (-diag_P + self.eps)
        W.diagonal().zero_()

        self.weight_matrix = W

        print("BSPM fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)

        return self.train_matrix_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
