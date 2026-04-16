import torch
import numpy as np
import scipy.sparse as sp
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy


class BSPM(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.K_b             = config['model'].get('K_b',              2)
        self.T_b             = config['model'].get('T_b',            1.0)
        self.K_s             = config['model'].get('K_s',              1)
        self.T_s             = config['model'].get('T_s',            2.5)
        self.idl_beta        = config['model'].get('idl_beta',       0.2)
        self.final_sharpening = config['model'].get('final_sharpening', True)
        self.eps             = 1e-12

    def fit(self, data_loader):
        print(f"Fitting BSPM (K_b={self.K_b}, T_b={self.T_b}, K_s={self.K_s}, T_s={self.T_s})...")

        X_sp = get_train_matrix_scipy(data_loader) # (U, I) sparse

        # ── 1. Blurring kernel G_b ──
        # G_b = D^{-1} (X^T X)
        print("  Computing blurring kernel G_b...")
        # X^T X on CPU
        G_raw_sp = X_sp.T @ X_sp # (I, I) sparse
        G_raw = G_raw_sp.toarray().astype(np.float32)
        
        D_b = G_raw.sum(axis=1, keepdims=True) + self.eps
        G_b = G_raw / D_b
        
        # ── 2. Combined weight matrix W ──
        # We compute the polynomial filter W such that Z_final = X @ W
        # Z_0 = I (identity item-item)
        print("  Computing combined polynomial filter W...")
        W = np.eye(self.n_items, dtype=np.float32)
        
        # Blurring phase: W = W @ ((1-h_b)I + h_b G_b)
        h_b = self.T_b / self.K_b
        W_blur = (1.0 - h_b) * np.eye(self.n_items, dtype=np.float32) + h_b * G_b
        for _ in range(self.K_b):
            W = W @ W_blur
            
        # Sharpening phase: W = W @ ((1+h_s)I - h_s G_b)
        h_s = self.T_s / self.K_s
        W_sharpen = (1.0 + h_s) * np.eye(self.n_items, dtype=np.float32) - h_s * G_b
        for _ in range(self.K_s):
            W = W @ W_sharpen
            
        # Final IDL sharpening
        if self.final_sharpening:
            W = W @ ((1.0 + self.idl_beta) * np.eye(self.n_items, dtype=np.float32) - self.idl_beta * G_b)

        self.weight_matrix = torch.tensor(W, dtype=torch.float32, device=self.device)
        self.train_matrix_gpu = self.get_train_matrix(data_loader)
        print("BSPM fitting complete.")

    def forward(self, user_indices):
        # S_hat = r_u @ W
        r_u = torch.index_select(self.train_matrix_gpu, 0, user_indices).to_dense()
        return r_u @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
