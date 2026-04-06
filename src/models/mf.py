import torch
import torch.nn as nn
from .base import BaseModel

class MF(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.emb_dim = config['model'].get('embedding_dim', 64)

        self.user_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.n_items, self.emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_indices):
        u_emb = self.user_emb(user_indices)
        return torch.matmul(u_emb, self.item_emb.weight.t())

    def calc_loss(self, batch_data):
        u_idx = batch_data['user_id'].squeeze()
        p_idx = batch_data['pos_item_id'].squeeze()
        n_idx = batch_data['neg_item_id'].squeeze()

        u_e = self.user_emb(u_idx)
        p_e = self.item_emb(p_idx)
        n_e = self.item_emb(n_idx)

        pos_scores = (u_e * p_e).sum(dim=-1)

        if n_e.dim() == 3:
            u_e_expanded = u_e.unsqueeze(1)
            neg_scores = (u_e_expanded * n_e).sum(dim=-1)
            loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores)).mean()
        else:
            neg_scores = (u_e * n_e).sum(dim=-1)
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

        return (loss,), None
