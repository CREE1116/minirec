import torch
import torch.nn as nn
from .base import BaseModel

class MF_IPS(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.emb_dim = config['model'].get('embedding_dim', 64)
        self.reg_weight = float(config['model'].get('reg_weight', 1e-4))
        self.beta = config['model'].get('beta', 0.5)

        self.user_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.n_items, self.emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        # Propensity based on item popularity: p_i ∝ n_i^beta
        item_counts = torch.FloatTensor(data_loader.item_popularity)
        propensity = torch.pow(item_counts, self.beta)
        propensity = propensity / propensity.max()
        # To avoid division by zero or very small values
        self.propensity = torch.clamp(propensity, min=1e-6).to(self.device)

    def forward(self, user_indices):
        u_emb = self.user_emb(user_indices)
        return torch.matmul(u_emb, self.item_emb.weight.t())

    def calc_loss(self, batch_data):
        u_idx = batch_data['user_id'].view(-1).to(self.device)
        p_idx = batch_data['pos_item_id'].view(-1).to(self.device)
        n_idx = batch_data['neg_item_id'].view(u_idx.shape[0], -1).to(self.device)

        u_e = self.user_emb(u_idx)
        p_e = self.item_emb(p_idx)
        n_e = self.item_emb(n_idx)

        pos_scores = (u_e * p_e).sum(dim=-1)
        neg_scores = (u_e.unsqueeze(1) * n_e).sum(dim=-1)

        # Propensities
        p_prop = self.propensity[p_idx]
        n_prop = self.propensity[n_idx]

        # IPS Weighted BPR Loss
        # L = - [ (1/p+) * log(sigmoid(pos-neg)) ]
        # Note: In pure IPS for MSE, it's (y - y_hat)^2 / p. 
        # For BPR, we weight the pairwise comparison.
        
        diff = pos_scores.unsqueeze(1) - neg_scores
        loss = -(torch.log(torch.sigmoid(diff) + 1e-10) / p_prop.unsqueeze(1)).mean()

        reg_loss = self.reg_weight * (
            u_e.pow(2).sum() + 
            p_e.pow(2).sum() + 
            n_e.pow(2).sum()
        ) / (2.0 * float(u_idx.shape[0]))

        return (loss, reg_loss), {'ips_bpr_loss': loss.item(), 'reg_loss': reg_loss.item()}
