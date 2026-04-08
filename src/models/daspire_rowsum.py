import torch
from .base import BaseModel

class DAspireRowsum(BaseModel):
    """Ablation: DAspire (DLAE-style) logic using only Rowsum (S_i) instead of SNR."""
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 100.0)
        self.alpha      = config['model'].get('alpha', 1.0)
        self.dropout_p  = config['model'].get('dropout_p', 0.5)
        self.eps        = 1e-12
        self.weight_matrix = None
        self.train_matrix  = None

    def fit(self, data_loader):
        print(f"Fitting DAspireRowsum (DLAE-style, S_i scaling), lambda={self.reg_lambda}, alpha={self.alpha}, p={self.dropout_p})")
        X = self.get_train_matrix(data_loader)
        self.train_matrix = X
        G = torch.sparse.mm(X.t(), X.to_dense()).to(self.device)
        S = G.sum(dim=1)

        # reliability = S / GM(S)
        log_S = torch.log(S + self.eps)
        reliability = (log_S - log_S.mean()).exp()

        scale_factor = torch.pow(reliability + self.eps, -self.alpha / 2.0)
        G_tilde = G * scale_factor.unsqueeze(1) * scale_factor.unsqueeze(0)

        p = min(self.dropout_p, 0.99)
        w = (p / (1.0 - p)) * G_tilde.diagonal()
        G_lhs = G_tilde.clone()
        G_lhs.diagonal().add_(w + self.reg_lambda)

        self.weight_matrix = torch.linalg.solve(G_lhs, G_tilde)

    def forward(self, user_indices):
        if not hasattr(self, 'train_matrix_dense'):
            self.train_matrix_dense = self.train_matrix.to_dense().to(self.device)
        return self.train_matrix_dense[user_indices] @ self.weight_matrix

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
