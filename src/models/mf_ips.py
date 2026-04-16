import torch
import torch.nn as nn
from .base import BaseModel

class MF_IPS(BaseModel):
    """
    Matrix Factorization with Inverse Propensity Scoring (MF-IPS).
    Strict Pointwise MSE version.
    
    The objective is to estimate the true risk by weighting observed samples:
    L = avg_{(u,i) \in D_pos} [ (1 - pred_ui)^2 / p_ui ] + avg_{(u,j) \in D_neg} [ (0 - pred_uj)^2 / p_uj ]
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.emb_dim = config['model'].get('embed_dim', 64)
        self.gamma = config['model'].get('gamma', 0.5)
        self.reg_weight = config['model'].get('reg_weight', 1e-3)

        self.user_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.n_items, self.emb_dim)

        # Normalized Propensity Scores
        item_counts = torch.FloatTensor(data_loader.item_popularity)
        p = (item_counts / (item_counts.sum() + 1e-12)).pow(self.gamma)
        self.propensity = p.clamp(min=1e-4).to(self.device)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_indices):
        u_e = self.user_emb(user_indices)
        return torch.mm(u_e, self.item_emb.weight.t())

    def calc_loss(self, batch):
        u_idx = batch['user_id'].squeeze()
        p_idx = batch['pos_item_id'].squeeze()
        n_idx = batch['neg_item_id'].squeeze()

        u_e = self.user_emb(u_idx)
        p_e = self.item_emb(p_idx)
        n_e = self.item_emb(n_idx)

        # ── 1. Pointwise Prediction Scores ──
        pos_scores = (u_e * p_e).sum(dim=1)
        if n_idx.dim() > 1:
            neg_scores = (u_e.unsqueeze(1) * n_e).sum(dim=2)
        else:
            neg_scores = (u_e * n_e).sum(dim=1)

        # ── 2. Propensities ──
        p_prop = self.propensity[p_idx]
        n_prop = self.propensity[n_idx]

        # ── 3. Pointwise IPS-weighted MSE Loss ──
        # Both positive and negative samples are weighted by 1/p
        pos_loss = ((1.0 - pos_scores).pow(2) / p_prop).mean()
        neg_loss = (neg_scores.pow(2) / n_prop).mean()

        total_loss = pos_loss + neg_loss

        # ── 4. Regularization ──
        reg_loss = self.reg_weight * (
            u_e.pow(2).sum() + p_e.pow(2).sum() + n_e.pow(2).sum()
        ) / u_idx.shape[0]

        return (total_loss, reg_loss), {
            'pos_ips_mse': pos_loss.item(),
            'neg_ips_mse': neg_loss.item(),
            'reg_loss': reg_loss.item()
        }
