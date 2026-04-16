import torch
import torch.nn as nn
from .base import BaseModel

class DR_JL(BaseModel):
    """
    True Doubly Robust Joint Learning (DR-JL).
    Strict Pointwise version based on "Doubly Robust Joint Learning for Recommendation".
    
    The target model is optimized to minimize the DR risk estimator:
    L_dr = avg [ (pred - dr_estimator)^2 ]
    where dr_estimator = (label - imp) / prop + imp
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.emb_dim = config['model'].get('embed_dim', 64)
        self.gamma = config['model'].get('gamma', 0.5)     # Propensity exponent
        self.eta = config['model'].get('eta', 1.0)         # Imputation loss weight
        self.reg_weight = config['model'].get('reg_weight', 1e-3)

        # 1. Prediction Model (The "Target" model)
        self.user_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.n_items, self.emb_dim)
        
        # 2. Imputation Model (Estimates the missing labels)
        self.user_emb_imp = nn.Embedding(self.n_users, self.emb_dim)
        self.item_emb_imp = nn.Embedding(self.n_items, self.emb_dim)

        # 3. Propensity Scores (Normalized and Clamped)
        item_counts = torch.FloatTensor(data_loader.item_popularity)
        p = (item_counts / (item_counts.sum() + 1e-12)).pow(self.gamma)
        self.propensity = p.clamp(min=1e-4).to(self.device)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.user_emb_imp.weight, std=0.01)
        nn.init.normal_(self.item_emb_imp.weight, std=0.01)

    def forward(self, user_indices):
        u_e = self.user_emb(user_indices)
        return torch.mm(u_e, self.item_emb.weight.t())

    def calc_loss(self, batch):
        u_idx = batch['user_id'].squeeze()
        p_idx = batch['pos_item_id'].squeeze()
        n_idx = batch['neg_item_id'].squeeze() 

        # embeddings
        u_e = self.user_emb(u_idx); p_e = self.item_emb(p_idx); n_e = self.item_emb(n_idx)
        u_e_imp = self.user_emb_imp(u_idx); p_e_imp = self.item_emb_imp(p_idx); n_e_imp = self.item_emb_imp(n_idx)

        # ── 1. Pointwise Prediction Scores ──
        pos_pred = (u_e * p_e).sum(dim=1)
        if n_idx.dim() > 1:
            neg_pred = (u_e.unsqueeze(1) * n_e).sum(dim=2)
            pos_imp = (u_e_imp * p_e_imp).sum(dim=1)
            neg_imp = (u_e_imp.unsqueeze(1) * n_e_imp).sum(dim=2)
            p_prop = self.propensity[p_idx].unsqueeze(1)
            n_prop = self.propensity[n_idx]
        else:
            neg_pred = (u_e * n_e).sum(dim=1)
            pos_imp = (u_e_imp * p_e_imp).sum(dim=1)
            neg_imp = (u_e_imp * n_e_imp).sum(dim=1)
            p_prop = self.propensity[p_idx]
            n_prop = self.propensity[n_idx]

        # ── 2. True DR Estimators (Targets) ──
        # dr = (label - imp) / prop + imp
        # We detach imputation to avoid direct gradient leakage into the imputation model during target optimization
        dr_pos_target = (1.0 - pos_imp.detach()) / p_prop + pos_imp.detach()
        dr_neg_target = (0.0 - neg_imp.detach()) / n_prop + neg_imp.detach()

        # ── 3. DR Pointwise Loss (Regression) ──
        dr_loss_pos = (pos_pred - dr_pos_target).pow(2).mean()
        dr_loss_neg = (neg_pred - dr_neg_target).pow(2).mean()
        dr_loss = dr_loss_pos + dr_loss_neg

        # ── 4. Imputation Model Supervision (MSE on observed labels) ──
        imp_loss_pos = (pos_imp - 1.0).pow(2).mean()
        imp_loss_neg = (neg_imp - 0.0).pow(2).mean()
        imp_loss = imp_loss_pos + imp_loss_neg

        # ── 5. Regularization ──
        reg_loss = self.reg_weight * (
            u_e.pow(2).sum() + p_e.pow(2).sum() + n_e.pow(2).sum() +
            u_e_imp.pow(2).sum() + p_e_imp.pow(2).sum() + n_e_imp.pow(2).sum()
        ) / u_idx.shape[0]

        total_loss = dr_loss + self.eta * imp_loss + reg_loss

        return (total_loss, reg_loss), {
            'dr_mse_loss': dr_loss.item(),
            'imp_mse_loss': imp_loss.item(),
            'reg_loss': reg_loss.item()
        }
