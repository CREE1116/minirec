import torch
import numpy as np
import gc
from .base import BaseModel
from src.utils.sparse import get_train_matrix_scipy

class IPS_LAE(BaseModel):
    """
    IPS_LAE: Inverse Propensity Scored Linear AutoEncoder
    Restored to Simple CLAE Style for Maximum Stability.
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = np.float32(config['model'].get('reg_lambda', 500.0))
        self.wbeta = np.float32(config['model'].get('wbeta', 0.4))
        self.wtype = config['model'].get('wtype', 'logsigmoid')
        self.eps = np.float32(1e-12)
        self.weight_matrix = None

    def _compute_inv_propensity(self, X):
        pop = np.array(X.sum(axis=0)).flatten().astype(np.float32)
        if self.wtype == 'powerlaw':
            p = np.power(pop / (np.max(pop) + self.eps), self.wbeta)
        elif self.wtype == 'logsigmoid':
            log_freqs = np.log(pop + 1.0)
            alpha_logit = -self.wbeta * (np.min(log_freqs) + np.max(log_freqs)) / 2.0
            p = 1.0 / (1.0 + np.exp(-(alpha_logit + self.wbeta * log_freqs)))
        else:
            p = np.ones_like(pop)
        return torch.tensor(1.0 / (p + self.eps), dtype=torch.float32, device=self.device)

    def fit(self, data_loader):
        print(f"Fitting IPS_LAE (lambda={self.reg_lambda}) using Simple CLAE style...")
        
        # 1. Get raw sparse matrix
        X = get_train_matrix_scipy(data_loader)
        self.train_matrix_cpu = X.tocsr()

        # 2. Gram Matrix (Direct simple way like CLAE)
        print("  computing gram matrix...")
        G = (X.T @ X).toarray().astype(np.float32)
        G[np.diag_indices(X.shape[1])] += self.reg_lambda
        
        # 3. Inversion (The Peak Point)
        print("  inverting matrix (NumPy)...")
        gc.collect() # Clear all sparse multiplication debris
        P = np.linalg.inv(G)
        del G # Delete input immediately
        gc.collect()
        
        # 4. Deriving Weights
        diag_P = np.diag(P).astype(np.float32)
        W_np = (-P / (diag_P + self.eps)).astype(np.float32)
        del P, diag_P
        np.fill_diagonal(W_np, 0)
        
        # 5. Move to GPU & IPS Weighting
        print("  applying IPS weighting on GPU...")
        self.weight_matrix = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        del W_np
        gc.collect()
        
        inv_p = self._compute_inv_propensity(X)
        self.weight_matrix *= inv_p.view(1, -1) # In-place multiplication
        self.weight_matrix.diagonal().zero_()
        
        del inv_p
        gc.collect()
        if 'cuda' in str(self.device): torch.cuda.empty_cache()
        print("IPS_LAE fitting complete.")

    def forward(self, user_indices):
        return self._get_batch_ratings(user_indices, self.weight_matrix)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
