import torch
import numpy as np
from tqdm import tqdm
import sys
import pandas as pd
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader

# Precomputed 1/log2(rank+2) and cumulative IDCG table (up to K=10000)
_LOG2_RECIP  = (1.0 / np.log2(np.arange(2, 10002, dtype=np.float64))).astype(np.float32)
_IDCG_TABLE  = np.concatenate([[np.float32(0.0)], np.cumsum(_LOG2_RECIP)])  # float32, index by n_relevant


# ── Diversity / fairness helpers (GPU-optimized) ──────────────

def get_gini_index_gpu(counts_tensor):
    """Calculate Gini Index directly on GPU."""
    if counts_tensor.sum() == 0: return 0.0
    n = counts_tensor.size(0)
    values, _ = torch.sort(counts_tensor)
    idx = torch.arange(1, n + 1, device=counts_tensor.device, dtype=torch.float32)
    gini = torch.sum((2 * idx - n - 1) * values) / (n * values.sum())
    return gini.item()

def get_novelty_gpu(counts_tensor, item_popularity_tensor, num_users):
    """Calculate Novelty directly on GPU."""
    if counts_tensor.sum() == 0: return 0.0
    # p_i: probability of item i being in training set
    # We use training popularity to define novelty
    total_train = item_popularity_tensor.sum()
    p_i = (item_popularity_tensor + 1) / (total_train + item_popularity_tensor.size(0))
    self_info = -torch.log2(p_i + 1e-10)
    
    # Weight self-information by recommendation frequency
    novelty = torch.sum(counts_tensor * self_info) / counts_tensor.sum()
    return novelty.item()

def get_item_split_sets(item_popularity, head_ratio=0.8, mid_ratio=0.1):
    """Split items into Head, Mid, and Tail based on interaction volume."""
    pop = pd.Series(item_popularity) if isinstance(item_popularity, np.ndarray) else item_popularity
    sorted_pop = pop.sort_values(ascending=False)
    total_vol = sorted_pop.sum()
    cumsum_vol = sorted_pop.cumsum().values
    
    head_cutoff = total_vol * head_ratio
    mid_cutoff = total_vol * (head_ratio + mid_ratio)
    
    head_n = int(np.searchsorted(cumsum_vol, head_cutoff, side='right'))
    mid_n = int(np.searchsorted(cumsum_vol, mid_cutoff, side='right'))
    
    head_items = sorted_pop.index[:head_n].tolist()
    mid_items = sorted_pop.index[head_n:mid_n].tolist()
    tail_items = sorted_pop.index[mid_n:].tolist()
    
    return head_items, mid_items, tail_items


# ── Core evaluation (Full-GPU Optimized) ─────────────────────────────────────────────

def _evaluate_full(model, test_loader, top_k_list, metrics_list, device,
                   user_history_sp, test_gt_sp, item_popularity=None, head_ratio=0.8, mid_ratio=0.1):
    """
    Vectorized evaluation using GPU Sparse Tensors.
    - user_history_sp: Sparse CSR tensor (n_users, n_items) on device
    - test_gt_sp: Sparse CSR tensor (n_users, n_items) on device
    """
    n_users = test_gt_sp.size(0)
    n_items = model.n_items
    max_k   = max(top_k_list)

    # Precompute weights on device
    log2_w    = torch.tensor(_LOG2_RECIP[:max_k], dtype=torch.float32, device=device)
    idcg_lut  = torch.tensor(_IDCG_TABLE[:max_k + 1], dtype=torch.float32, device=device)

    # Precompute popularity tensor for PopRatio
    pop_tensor = None
    mean_pop = 1.0
    if 'PopRatio' in metrics_list and item_popularity is not None:
        pop_tensor = torch.tensor(item_popularity, dtype=torch.float, device=device)
        mean_pop   = pop_tensor.mean().item()

    # Group Masks (Head, Mid, Tail)
    group_masks = {}
    needed_groups = [g for g in ['Head', 'Mid', 'Tail'] if any(m.startswith(g) for m in metrics_list) or (g == 'Tail' and 'LongTail' in str(metrics_list))]
    
    if item_popularity is not None and needed_groups:
        h_idx, m_idx, t_idx = get_item_split_sets(item_popularity, head_ratio, mid_ratio)
        for g, idxs in zip(['Head', 'Mid', 'Tail'], [h_idx, m_idx, t_idx]):
            mask = torch.zeros(n_items, dtype=torch.bool, device=device)
            mask[torch.tensor(idxs, dtype=torch.long, device=device)] = True
            group_masks[g] = mask

    # Accumulators
    sums = defaultdict(float)
    counts = defaultdict(float)
    # rec_counts per K for strict diversity metrics
    rec_counts_dict = {k: torch.zeros(n_items, device=device, dtype=torch.float32) for k in top_k_list}

    batch_size = test_loader.batch_size
    unique_users = torch.arange(n_users, device=device)

    with torch.no_grad():
        for start in tqdm(range(0, n_users, batch_size), desc="Eval (GPU)", file=sys.stdout):
            u_ids = unique_users[start : start + batch_size]
            B = u_ids.size(0)

            # 1. Forward Pass
            scores = model.forward(u_ids)  # (B, n_items)

            # 2. Vectorized Masking (Mask seen items)
            history_batch = torch.index_select(user_history_sp, 0, u_ids).to_dense() > 0
            scores[history_batch] = -1e10

            # 3. Get Top-K
            _, top_idx = torch.topk(scores, k=max_k, dim=1)
            
            # 4. Vectorized Ground-Truth Check
            gt_batch = torch.index_select(test_gt_sp, 0, u_ids).to_dense() > 0
            gt_sizes = gt_batch.sum(dim=1).float()
            
            # hit_mat: 1 if recommended item is in GT
            hit_mat = torch.gather(gt_batch.float(), 1, top_idx)

            for k in top_k_list:
                # Update rec_counts strictly for this K
                current_top_k = top_idx[:, :k].flatten()
                rec_counts_dict[k].scatter_add_(0, current_top_k, torch.ones(current_top_k.size(0), device=device))

                hk = hit_mat[:, :k]
                hit_sum = hk.sum(dim=1)
                n_rel = gt_sizes.clamp(max=k).long()

                if 'HitRate' in metrics_list: sums[f'HitRate@{k}'] += (hit_sum > 0).float().sum().item()
                if 'Recall' in metrics_list: sums[f'Recall@{k}'] += (hit_sum / gt_sizes.clamp(min=1)).sum().item()
                if 'Precision' in metrics_list: sums[f'Precision@{k}'] += (hit_sum / k).sum().item()
                if 'NDCG' in metrics_list:
                    dcg = (hk * log2_w[:k]).sum(dim=1)
                    idcg = idcg_lut[n_rel]
                    sums[f'NDCG@{k}'] += (dcg / idcg.clamp(min=1e-8)).sum().item()

                # Group metrics (Head/Mid/Tail)
                for g in ['Head', 'Mid', 'Tail']:
                    mask = group_masks.get(g)
                    if mask is not None:
                        # hit_mat for group g
                        g_gt_batch = gt_batch & mask
                        g_sz = g_gt_batch.sum(dim=1).float()
                        g_hit = torch.gather(g_gt_batch.float(), 1, top_idx[:, :k])
                        
                        valid_mask = g_sz > 0
                        n_valid = valid_mask.sum().item()
                        if n_valid > 0:
                            h_v = g_hit[valid_mask]
                            s_v = g_sz[valid_mask]
                            h_sum = h_v.sum(dim=1)
                            
                            tags = [g]
                            if g == 'Tail': tags.append('LongTail')
                            
                            for tag in tags:
                                if f'{tag}HitRate' in metrics_list:
                                    sums[f'{tag}HitRate@{k}'] += (h_sum > 0).float().sum().item()
                                    counts[f'{tag}HitRate@{k}'] += n_valid
                                if f'{tag}Recall' in metrics_list:
                                    sums[f'{tag}Recall@{k}'] += (h_sum / s_v.clamp(min=1)).sum().item()
                                    counts[f'{tag}Recall@{k}'] += n_valid
                                if f'{tag}NDCG' in metrics_list:
                                    dcg_t = (h_v * log2_w[:k]).sum(dim=1)
                                    idcg_t = idcg_lut[s_v.clamp(max=k).long()]
                                    sums[f'{tag}NDCG@{k}'] += (dcg_t / idcg_t.clamp(min=1e-8)).sum().item()
                                    counts[f'{tag}NDCG@{k}'] += n_valid

                if pop_tensor is not None:
                    sums[f'PopRatio@{k}'] += (pop_tensor[top_idx[:, :k]].mean(dim=1) / mean_pop).sum().item()

                counts[k] += B

    final = {}
    for k in top_k_list:
        n = counts[k]
        for m in ['HitRate', 'Recall', 'Precision', 'NDCG']:
            key = f'{m}@{k}'
            if key in sums: final[key] = sums[key] / n if n else 0.0
        
        for g in ['Head', 'Mid', 'Tail', 'LongTail']:
            for m in ['HitRate', 'Recall', 'NDCG']:
                key = f'{g}{m}@{k}'
                if key in sums: final[key] = sums[key] / counts.get(key, 1)

    return final, rec_counts_dict, group_masks


def evaluate_metrics(model, data_loader, eval_config, device, test_loader, is_final=False):
    top_k_list  = eval_config.get('top_k', [10])
    metrics_list = eval_config.get('metrics', ['NDCG', 'Recall'])
    history_dict = data_loader.eval_user_history if is_final else data_loader.train_user_history
    item_pop    = data_loader.item_popularity
    n_users     = data_loader.n_users
    n_items     = data_loader.n_items

    # ── 1. Build Sparse History & GT Matrices on GPU ──────────────────────────
    def build_sparse_matrix(data_dict_or_loader, rows, cols):
        if isinstance(data_dict_or_loader, dict):
            # history_dict: user_id -> set([item_id, ...])
            r, c = [], []
            for u, items in data_dict_or_loader.items():
                r.extend([u] * len(items))
                c.extend(list(items))
        else:
            # test_loader ground-truth
            dataset = data_dict_or_loader.dataset
            if isinstance(dataset, TensorDataset):
                r = dataset.tensors[0].numpy()
                c = dataset.tensors[1].numpy()
            else:
                us, it = [], []
                for u, i in data_dict_or_loader:
                    us.append(u.numpy()); it.append(i.numpy())
                r, c = np.concatenate(us), np.concatenate(it)
        
        i = torch.stack([torch.tensor(r, dtype=torch.long, device=device), 
                         torch.tensor(c, dtype=torch.long, device=device)])
        v = torch.ones(len(r), dtype=torch.float32, device=device)
        # Use Sparse COO for widest compatibility and index_select support
        return torch.sparse_coo_tensor(i, v, (rows, cols), device=device).coalesce()

    user_history_sp = build_sparse_matrix(history_dict, n_users, n_items)
    test_gt_sp      = build_sparse_matrix(test_loader, n_users, n_items)

    # ── 2. Run Evaluation Loop ────────────────────────────────────────────────
    final_results, rec_counts_dict, group_masks = _evaluate_full(
        model, test_loader, top_k_list, metrics_list, device, 
        user_history_sp, test_gt_sp, item_pop,
        eval_config.get('head_ratio', 0.8), eval_config.get('mid_ratio', 0.1)
    )

    # ── 3. Calculate Global Diversity Metrics ──────────────────────────────────
    if item_pop is not None:
        item_pop_t = torch.tensor(item_pop, dtype=torch.float32, device=device)
    
    for k in top_k_list:
        rec_counts = rec_counts_dict[k]
        
        if 'Coverage' in metrics_list:
            final_results[f'Coverage@{k}'] = (rec_counts > 0).float().mean().item()
        if 'GiniIndex' in metrics_list:
            final_results[f'GiniIndex@{k}'] = get_gini_index_gpu(rec_counts)
        if 'Novelty' in metrics_list and item_pop is not None:
            final_results[f'Novelty@{k}'] = get_novelty_gpu(rec_counts, item_pop_t, n_users)
            
        for g in ['Head', 'Mid', 'Tail', 'LongTail']:
            m_name = f'{g}Coverage'
            if m_name in metrics_list:
                mask = group_masks.get('Tail' if g == 'LongTail' else g)
                if mask is not None:
                    g_counts = rec_counts[mask]
                    final_results[f'{m_name}@{k}'] = (g_counts > 0).float().mean().item()

    return final_results
