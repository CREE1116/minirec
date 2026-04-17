import torch
import numpy as np
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix

class LAE(BaseModel):
    """
    LAE: Linear AutoEncoder (Wiener Filter form)
    Formula: S = G @ (G + lambda * I)^-1
    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.weight_matrix = None

    def fit(self, data_loader):
        print(f"Fitting LAE (lambda={self.reg_lambda}) on {self.device}...")
        
        # 1. Load data and store for hybrid inference
        X_sp = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X_sp.tocsr()

        print(f"  computing gram matrix (on {self.device})...")
        G_np = compute_gram_matrix(X_sp, data_loader, device=self.device)
        
        # 2. Solve linear system
        if 'cuda' in str(self.device) and G_np.shape[0] < 20000:
            print("  solving linear system (GPU)...")
            G_torch = torch.from_numpy(G_np).to(self.device)
            # A = G + lambda * I
            A_torch = G_torch.clone()
            A_torch.diagonal().add_(self.reg_lambda)
            
            try:
                self.weight_matrix = torch.linalg.solve(A_torch, G_torch)
            except RuntimeError:
                print("[Warning] Singular matrix, using stronger regularization.")
                A_torch.diagonal().add_(1e-4)
                self.weight_matrix = torch.linalg.solve(A_torch, G_torch)
            
            del G_torch, A_torch, G_np
        else:
            print("  solving linear system (CPU)...")
            # G_np is already a fresh copy from compute_gram_matrix
            A_np = G_np
            A_np[np.diag_indices_from(A_np)] += self.reg_lambda
            
            # G_np를 보존해야 하므로, compute_gram_matrix에서 하나 더 가져옴 (Sparse 캐시 활용)
            # 하지만 메모리 아끼기 위해 A_np 수정 전의 G를 다시 확보하는 대신...
            # A_np = G + lambda*I 이므로 G = A_np - lambda*I 임을 이용하거나
            # 그냥 원본을 다시 불러오기 (Sparse->Dense는 빠름)
            G_orig = compute_gram_matrix(X_sp, data_loader) 

            try:
                W_np = np.linalg.solve(A_np, G_orig)
            except np.linalg.LinAlgError:
                print("[Warning] Singular matrix, using stronger regularization.")
                A_np[np.diag_indices_from(A_np)] += 1e-4
                W_np = np.linalg.solve(A_np, G_orig)
            
            self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
            del A_np, G_orig, W_np, G_np

        gc.collect()
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        print("LAE fitting complete.")

    def forward(self, user_indices):
        """Memory-efficient hybrid inference"""
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
