import torch
import torch.nn as nn
from .base import BaseModel

class IPS_LAE(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.wbeta = config['model'].get('wbeta', 0.4)
        self.wtype = config['model'].get('wtype', 'logsigmoid')  # powerlaw | logsigmoid
        self.weight_matrix = None
        self.train_matrix = None

    def _compute_inv_propensity(self, X):
        pop = torch.sparse.sum(X, dim=0).to_dense()
        if self.wtype == 'powerlaw':
            norm_pop = pop / (torch.max(pop) + 1e-12)
            p = torch.pow(norm_pop, self.wbeta)
        elif self.wtype == 'logsigmoid':
            log_freqs = torch.log(pop + 1)
            alpha_logit = -self.wbeta * (torch.min(log_freqs) + torch.max(log_freqs)) / 2
            logits = alpha_logit + self.wbeta * log_freqs
            p = torch.sigmoid(logits)
        else:
            p = torch.ones_like(pop)
        return 1 / (p + 1e-12)

    def fit(self, data_loader):
        print(f"Fitting IPS_LAE (wtype={self.wtype}, wbeta={self.wbeta}) on {self.device}...")
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X

        G = torch.sparse.mm(X.t(), X).to_dense().to(self.device)
        G.diagonal().add_(self.reg_lambda)
        P = torch.linalg.inv(G)

        diag_P = P.diagonal()
        B = P / (-diag_P + 1e-12)
        B.diagonal().zero_()

        inv_p = self._compute_inv_propensity(X).to(self.device)
        B = B * inv_p
        self.weight_matrix = B
        print("IPS_LAE fitting complete.")

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)
        return self.train_matrix_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
