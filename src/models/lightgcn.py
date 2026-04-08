import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel

class LightGCN(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.emb_dim = config['model'].get('embedding_dim', 64)
        self.n_layers = config['model'].get('n_layers', 3)
        self.reg_weight = float(config['model'].get('reg_weight', 1e-4))

        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        self.norm_adj = self._get_norm_adj(data_loader)
        self.final_user_emb = None
        self.final_item_emb = None

    def _get_norm_adj(self, data_loader):
        # Adjacency is (n_users+n_items)^2 — keep sparse to avoid OOM
        adj = data_loader.get_interaction_graph(add_self_loops=False)
        if self.device.type == 'cuda':
            adj = adj.to(self.device)
        
        adj = adj.coalesce()
        row_sum = torch.sparse.sum(adj, dim=1).to_dense()
        d_inv = torch.pow(row_sum, -0.5)
        d_inv[torch.isinf(d_inv)] = 0.
        
        indices = adj.indices()
        values = adj.values() * d_inv[indices[0]] * d_inv[indices[1]]
        
        norm_adj = torch.sparse_coo_tensor(indices, values, adj.shape).coalesce()
        if self.device.type == 'cuda':
            norm_adj = norm_adj.to(self.device)
        return norm_adj

    def _propagate(self):
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            # MPS does not support sparse mm; fall back to CPU for this op
            if self.device.type == 'mps':
                all_emb = torch.sparse.mm(self.norm_adj, all_emb.cpu()).to(self.device)
            else:
                all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            embs.append(all_emb)
        final_emb = torch.mean(torch.stack(embs, dim=0), dim=0)
        return torch.split(final_emb, [self.n_users, self.n_items])

    def forward(self, user_indices):
        if self.training or self.final_user_emb is None:
            u_f, i_f = self._propagate()
        else:
            u_f, i_f = self.final_user_emb, self.final_item_emb
        return u_f[user_indices] @ i_f.t()

    def calc_loss(self, batch_data):
        u_f, i_f = self._propagate()

        u_idx = batch_data['user_id'].view(-1)
        p_idx = batch_data['pos_item_id'].view(-1)
        n_idx = batch_data['neg_item_id'].view(u_idx.shape[0], -1) 

        u_e, p_e = u_f[u_idx], i_f[p_idx] 
        n_e = i_f[n_idx]                  

        pos_scores = (u_e * p_e).sum(dim=-1, keepdim=True) 
        neg_scores = (u_e.unsqueeze(1) * n_e).sum(dim=-1)  

        # BPR Loss
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # L2 Regularization (Explicit tensor math on RAW embeddings)
        u_e_raw = self.user_embedding(u_idx)
        p_e_raw = self.item_embedding(p_idx)
        n_e_raw = self.item_embedding(n_idx)
        
        reg_loss = self.reg_weight * (
            u_e_raw.pow(2).sum() +
            p_e_raw.pow(2).sum() +
            n_e_raw.pow(2).sum()
        ) / (2.0 * float(u_idx.shape[0]))

        if not self.training:
            self.final_user_emb, self.final_item_emb = u_f.detach(), i_f.detach()

        return (loss, reg_loss), {'bpr_loss': loss.item(), 'reg_loss': reg_loss.item()}
