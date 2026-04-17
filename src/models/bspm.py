import torch
import numpy as np
import scipy.sparse as sp
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy, compute_gram_matrix


class BSPM(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.K_b             = config['model'].get('K_b', 2)
        self.T_b             = np.float32(config['model'].get('T_b', 1.0))
        self.K_s             = config['model'].get('K_s', 1)
        self.T_s             = np.float32(config['model'].get('T_s', 2.5))
        self.idl_beta        = np.float32(config['model'].get('idl_beta', 0.2))
        self.final_sharpening = config['model'].get('final_sharpening', True)
        self.eps             = np.float32(1e-12)

    def fit(self, data_loader):
        print(f"Fitting BSPM on CPU float32...")

        X_sp = get_train_matrix_scipy(data_loader) 
        self.train_matrix_cpu = X_sp.tocsr() # Hybrid inference

        # ── 1. Blurring kernel G_b ──
        # G_b = D^{-1} (X^T X)
        print("  Computing blurring kernel G_b (Block-wise)...")
        # X^T X via optimized utility
        G_raw = compute_gram_matrix(X_sp, data_loader)
        
        D_b = G_raw.sum(axis=1, keepdims=True).astype(np.float32) + self.eps
        G_b = (G_raw / D_b).astype(np.float32)
        del G_raw, D_b
        gc.collect()
        
        # ── 2. Combined weight matrix W ──
        print("  Computing combined polynomial filter W...")
        W = np.eye(self.n_items, dtype=np.float32)
        
        h_b = (self.T_b / np.float32(self.K_b)).astype(np.float32)
        W_blur = ((np.float32(1.0) - h_b) * np.eye(self.n_items, dtype=np.float32) + h_b * G_b).astype(np.float32)
        for _ in range(self.K_b):
            W = (W @ W_blur).astype(np.float32)
            
        del W_blur
        gc.collect()
            
        h_s = (self.T_s / np.float32(self.K_s)).astype(np.float32)
        W_sharpen = ((np.float32(1.0) + h_s) * np.eye(self.n_items, dtype=np.float32) - h_s * G_b).astype(np.float32)
        for _ in range(self.K_s):
            W = (W @ W_sharpen).astype(np.float32)
            
        del W_sharpen
        gc.collect()
            
        if self.final_sharpening:
            W = (W @ ((np.float32(1.0) + self.idl_beta) * np.eye(self.n_items, dtype=np.float32) - self.idl_beta * G_b)).astype(np.float32)

        self.weight_matrix = torch.tensor(W, dtype=torch.float32, device=self.device)
        
        del W, G_b
        gc.collect()
        
        print("BSPM fitting complete.")

    def forward(self, user_indices):
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
