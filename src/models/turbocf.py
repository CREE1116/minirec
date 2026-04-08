import torch
import numpy as np
from .base import BaseModel


class TurboCF(BaseModel):
    """
    TurboCF: Matrix Decomposition-Free Graph Filtering
    (Park et al., SIGIR 2024)

    핵심 아이디어:
        행렬 분해나 명시적인 가중치 행렬(W) 생성 없이, 
        Sparse한 Gram 행렬 위에서 다항식 필터를 직접 적용하여 추천 신호를 추출합니다.
        
    수정 사항:
        - 모든 연산을 Sparse 상태로 유지하여 메모리 효율 극대화
        - Explicit W 행렬을 만들지 않고 Forward 시점에 Polynomial Filtering 수행
        - EASE 필터 함수(mu / (mu + lambda))를 Chebyshev 다항식으로 근사
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.K          = config['model'].get('K', 3)        # Polynomial degree
        self.alpha      = config['model'].get('alpha', 0.5)  # Normalization exponent
        self.eps        = 1e-12
        
        # 가중치 행렬 대신 필터 계수와 정규화된 Gram 행렬(Sparse)을 저장
        self.coeffs = None
        self.G_tilde = None
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting TurboCF (lambda={self.reg_lambda}, K={self.K}, "
              f"alpha={self.alpha}) on {self.device}...")
        
        # X: (Users x Items) Sparse Tensor
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X # Sparse 유지

        # ── 1. Sparse Gram matrix G = X^T X ────────────────────────────────
        # Dense 변환 없이 순수 Sparse 연산 수행
        G = torch.sparse.mm(X.t(), X).coalesce()
        n_i = torch.sparse.sum(G, dim=1).to_dense() # 각 아이템의 차수

        # ── 2. Sparse Symmetric Normalization ─────────────────────────────
        # G_tilde = D^{-alpha/2} G D^{-alpha/2}
        inv_sqrt = torch.pow(n_i + self.eps, -self.alpha / 2.0)
        
        indices = G.indices()
        rows, cols = indices[0], indices[1]
        vals = G.values()
        
        # Sparse 원소별로 정규화 적용
        vals_norm = vals * inv_sqrt[rows] * inv_sqrt[cols]
        G_tilde = torch.sparse_coo_tensor(indices, vals_norm, G.shape).coalesce().to(self.device)
        self.G_tilde = G_tilde

        # ── 3. Polynomial Filter via Chebyshev Approximation ───────────────
        # 고유값 범위 추정 (Sparse mv 사용)
        mu_max = self._estimate_max_eigenvalue(G_tilde)

        # EASE 필터 c_k 계수 산출
        self.coeffs = self._ease_chebyshev_coeffs(
            K=self.K,
            lam=self.reg_lambda,
            mu_max=mu_max
        ).to(self.device)

        print(f"TurboCF fitting complete. mu_max: {mu_max:.4f}")

    def forward(self, user_indices):
        # ── 4. Explicit W 없이 Polynomial Filtering 수행 ──────────────────
        # W = Σ c_k G_tilde^k
        # y_hat = X @ W = Σ c_k (X @ G_tilde^k)
        
        # 속도를 위해 학습 행렬을 dense로 캐싱 (EASE와 동일)
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)
            
        # Z: (Batch x Items) Dense Tensor
        Z = self.train_matrix_dense[user_indices]
        out = self.coeffs[0] * Z
        
        # G_tilde (Sparse)와 Z (Dense)의 곱셈 수행
        # Z @ G_tilde = (G_tilde @ Z.T).T  (Dense-Sparse MM은 이 방식이 가장 효율적)
        for k in range(1, self.K + 1):
            # Z_next = Z @ G_tilde
            Z = torch.sparse.mm(self.G_tilde, Z.t()).t()
            out = out + self.coeffs[k] * Z
            
        return out

    def _estimate_max_eigenvalue(self, G, n_iter=20):
        """Power iteration using sparse matrix-matrix multiplication."""
        v = torch.randn(G.shape[0], device=self.device)
        v = v / v.norm()
        mu = 1.0
        for _ in range(n_iter):
            # torch.sparse.mv 대신 mm 사용 (vector를 N x 1 matrix로 취급)
            v = torch.sparse.mm(G, v.unsqueeze(1)).squeeze()
            mu = v.norm()
            v = v / (mu + self.eps)
        return float(mu)

    def _ease_chebyshev_coeffs(self, K, lam, mu_max):
        """Compute Chebyshev coefficients for the EASE filter function."""
        import numpy as np
        n_points = max(100, K * 10)
        idx = np.arange(1, n_points + 1)
        theta = np.pi * (2 * idx - 1) / (2 * n_points)
        mu_tilde = np.cos(theta)
        mu = (mu_tilde + 1) / 2 * mu_max
        f_vals = mu / (mu + lam)

        coeffs = []
        for k in range(K + 1):
            T_k = np.cos(k * theta)
            c_k = (2.0 / n_points) * np.sum(f_vals * T_k)
            if k == 0:
                c_k /= 2.0
            coeffs.append(float(c_k))

        return torch.tensor(coeffs, dtype=torch.float32)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
