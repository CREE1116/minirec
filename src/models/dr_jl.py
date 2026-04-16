import torch
import torch.nn as nn
from .base import BaseModel

class DR_JL(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.emb_dim = config['model'].get('embedding_dim', 64)
        self.reg_weight = float(config['model'].get('reg_weight', 1e-4))
        self.beta = config['model'].get('beta', 0.5)

        # Prediction Model
        self.user_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.n_items, self.emb_dim)
        
        # Imputation Model
        self.user_emb_imp = nn.Embedding(self.n_users, self.emb_dim)
        self.item_emb_imp = nn.Embedding(self.n_items, self.emb_dim)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.user_emb_imp.weight, std=0.01)
        nn.init.normal_(self.item_emb_imp.weight, std=0.01)

        # Propensity
        item_counts = torch.FloatTensor(data_loader.item_popularity)
        propensity = torch.pow(item_counts, self.beta)
        propensity = propensity / propensity.max()
        self.propensity = torch.clamp(propensity, min=1e-6).to(self.device)

    def forward(self, user_indices):
        u_emb = self.user_emb(user_indices)
        return torch.matmul(u_emb, self.item_emb.weight.t())

    def calc_loss(self, batch_data):
        u_idx = batch_data['user_id'].view(-1).to(self.device)
        p_idx = batch_data['pos_item_id'].view(-1).to(self.device)
        n_idx = batch_data['neg_item_id'].view(u_idx.shape[0], -1).to(self.device)

        # --- Prediction Model ---
        u_e = self.user_emb(u_idx)
        p_e = self.item_emb(p_idx)
        n_e = self.item_emb(n_idx)
        
        pos_scores = (u_e * p_e).sum(dim=-1)
        neg_scores = (u_e.unsqueeze(1) * n_e).sum(dim=-1)

        # --- Imputation Model ---
        u_e_imp = self.user_emb_imp(u_idx)
        p_e_imp = self.item_emb_imp(p_idx)
        n_e_imp = self.item_emb_imp(n_idx)

        pos_imp = (u_e_imp * p_e_imp).sum(dim=-1)
        neg_imp = (u_e_imp.unsqueeze(1) * n_e_imp).sum(dim=-1)

        # --- Losses ---
        p_prop = self.propensity[p_idx]
        
        # 1. IPS Loss (Observed)
        diff = pos_scores.unsqueeze(1) - neg_scores
        ips_loss = -(torch.log(torch.sigmoid(diff) + 1e-10) / p_prop.unsqueeze(1)).mean()

        # 2. Imputation Loss (All pairs in batch)
        # We want imputation model to predict the error or the score.
        # In DR-JL for implicit, we often use it to impute labels.
        # Here we use a simplified version where imputation model learns to reduce variance.
        imp_loss_pos = ((pos_imp - pos_scores.detach()) ** 2).mean()
        imp_loss_neg = ((neg_imp - neg_scores.detach()) ** 2).mean()
        imp_loss = imp_loss_pos + imp_loss_neg

        # 3. DR refinement: pred_model also tries to match imputation on unobserved
        # (Simplified implementation of DR principle)
        dr_loss = ((pos_scores - pos_imp.detach()) ** 2).mean() + ((neg_scores - neg_imp.detach()) ** 2).mean()

        reg_loss = self.reg_weight * (
            u_e.pow(2).sum() + p_e.pow(2).sum() + n_e.pow(2).sum() +
            u_e_imp.pow(2).sum() + p_e_imp.pow(2).sum() + n_e_imp.pow(2).sum()
        ) / (2.0 * float(u_idx.shape[0]))

        total_loss = ips_loss + imp_loss + dr_loss

        return (total_loss, reg_loss), {
            'ips_loss': ips_loss.item(),
            'imp_loss': imp_loss.item(),
            'dr_loss': dr_loss.item(),
            'reg_loss': reg_loss.item()
        }
