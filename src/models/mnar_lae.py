import torch
from .base import BaseModel


class MNAR_LAE(BaseModel):
    """
    Improved MNAR-LAE with Log-Sigmoid Propensity Estimation
    
    Philosophy:
    1. Initial propensity estimation using a centered log-sigmoid function.
    2. Adaptive regularization to protect long-tail items.
    3. Self-consistent loop with 0.2 damping to reach a robust equilibrium.
    
    Proposed by: Keuri (크리)
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.lambda_0 = config['model'].get('reg_lambda', 100.0)
        self.gamma = config['model'].get('gamma', 2.0)    # Sigmoid slope (beta in snippet)
        self.beta = config['model'].get('beta', 0.5)      # Adaptive reg exponent
        self.alpha = config['model'].get('alpha', 0.2)   # Damping factor (mixing ratio)
        self.max_iter = config['model'].get('max_iter', 3) # 3 iterations are usually enough
        self.tol = config['model'].get('tol', 1e-4)
        self.eps = 1e-8
        
        self.weight_matrix = None
        self.propensity = None
        self.train_matrix = None

    def fit(self, data_loader):
        print(f"Fitting Improved MNAR-LAE (Log-Sigmoid) on {self.device}...")
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X
        X_dense = X.to_dense().to(self.device)
        num_items = X_dense.size(1)

        # 1. 초기 성향 추정 (Log-Sigmoid 기반)
        n_i = torch.sum(X_dense, dim=0)
        log_n = torch.log(n_i + 1)
        
        # Centering logic from snippet
        mid_log = (torch.min(log_n) + torch.max(log_n)) / 2
        alpha_centering = -self.gamma * torch.log(mid_log + self.eps)
        
        p = 1.0 / (1.0 + torch.exp(-alpha_centering - self.gamma * log_n))
        self.propensity = torch.clamp(p, min=self.eps, max=1.0 - self.eps)

        for i in range(self.max_iter):
            p_old = self.propensity.clone()

            # --- A. 현재 p를 바탕으로 모델(B) 학습 ---
            inv_p = 1.0 / (self.propensity + self.eps)
            X_tilde = X_dense * inv_p.unsqueeze(0)
            C_hat = torch.mm(X_tilde.t(), X_tilde)

            # 적응형 규제 계산
            lambda_i = self.lambda_0 / (torch.pow(self.propensity, self.beta) + self.eps)
            A = C_hat.clone()
            idx = torch.arange(num_items, device=self.device)
            A[idx, idx] += lambda_i
            
            # EASE 최적해 도출
            try:
                P_mat = torch.inverse(A)
                self.weight_matrix = - P_mat / (torch.diag(P_mat) + self.eps)
                self.weight_matrix[idx, idx] = 0.0
            except (torch._C._LinAlgError, RuntimeError):
                A.diagonal().add_(self.lambda_0)
                P_mat = torch.inverse(A)
                self.weight_matrix = - P_mat / (torch.diag(P_mat) + self.eps)
                self.weight_matrix[idx, idx] = 0.0

            # --- B. 학습된 모델(B)로부터 새로운 성향(p_new) 도출 ---
            with torch.no_grad():
                scores = torch.mm(X_tilde, self.weight_matrix)
                # Softmax mean as new propensity estimate
                p_new = torch.mean(torch.softmax(scores, dim=1), dim=0)
                p_new = p_new / (p_new.max() + self.eps)
            
            # --- C. 성향 업데이트 (Damping: 0.8 * old + 0.2 * new) ---
            # self.alpha default is 0.2
            self.propensity = (1 - self.alpha) * self.propensity + self.alpha * p_new

            # --- D. 수렴 체크 ---
            diff = torch.norm(self.propensity - p_old)
            print(f"[Iter {i+1}] Propensity Change: {diff.item():.6f}")
            if diff < self.tol:
                break

        print("Improved MNAR-LAE fitting complete.")

    def forward(self, user_indices):
        if self.train_matrix is None:
            raise RuntimeError("Model must be fitted before calling forward")
            
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)
            
        # 인퍼런스 시에도 학습된 최종 성향으로 입력 보정
        X_user = self.train_matrix_dense[user_indices]
        X_tilde = X_user / (self.propensity.unsqueeze(0) + self.eps)
        
        return torch.mm(X_tilde, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
