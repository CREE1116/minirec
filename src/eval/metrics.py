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


# ── Diversity / fairness helpers (called once, not in eval loop) ──────────────

def get_coverage(flat_recs, n_items):
    return len(set(flat_recs)) / n_items if n_items else 0.0

def get_gini_index(values):
    if len(values) == 0: return 0.0
    values = np.sort(values)
    n = len(values)
    if values.sum() == 0: return 0.0
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * values)) / (n * values.sum())

def get_gini_index_from_recs(flat_recs, n_items):
    counts = np.zeros(n_items)
    for item in flat_recs:
        if item < n_items: counts[item] += 1
    return get_gini_index(counts)

def get_novelty(flat_recs, item_popularity, num_users=None):
    if not flat_recs: return 0.0
    recs = np.array(flat_recs)
    pop = np.array(item_popularity) if not isinstance(item_popularity, np.ndarray) else item_popularity
    total = pop.sum() if num_users is None else num_users
    p_i = (pop[recs] + 1) / (total + len(pop))
    return float(np.mean(-np.log2(p_i + 1e-10)))

def get_item_split_sets(item_popularity, head_ratio=0.8, mid_ratio=0.1):
    """
    Split items into Head, Mid, and Tail based on interaction volume.
    - Head: Top items covering head_ratio of interactions.
    - Mid: Next items covering mid_ratio of interactions.
    - Tail: Remaining items.
    """
    pop = pd.Series(item_popularity) if isinstance(item_popularity, np.ndarray) else item_popularity
    sorted_pop = pop.sort_values(ascending=False)
    total_vol = sorted_pop.sum()
    cumsum_vol = sorted_pop.cumsum().values
    
    head_cutoff = total_vol * head_ratio
    mid_cutoff = total_vol * (head_ratio + mid_ratio)
    
    head_n = int(np.searchsorted(cumsum_vol, head_cutoff, side='right'))
    mid_n = int(np.searchsorted(cumsum_vol, mid_cutoff, side='right'))
    
    head_items = set(sorted_pop.index[:head_n].tolist())
    mid_items = set(sorted_pop.index[head_n:mid_n].tolist())
    tail_items = set(sorted_pop.index[mid_n:].tolist())
    
    return head_items, mid_items, tail_items

def get_long_tail_item_set(item_popularity, head_volume_percent=0.8):
    # Deprecated but kept for backward compatibility if needed
    _, _, tail = get_item_split_sets(item_popularity, head_volume_percent, 1.0 - head_volume_percent)
    return tail

def get_group_coverage(flat_recs, group_set):
    if not group_set: return 0.0
    return len(set(flat_recs) & group_set) / len(group_set)

def get_entropy_from_recs(flat_recs):
    if not flat_recs: return 0.0
    counts = np.array(list(defaultdict(int, {i: flat_recs.count(i) for i in set(flat_recs)}).values()), dtype=float)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-10)))

def get_ild(all_top_k_items, item_embeddings):
    scores = []
    for recs in all_top_k_items:
        if len(recs) < 2: continue
        emb = item_embeddings[recs]
        emb = emb / emb.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        sim = torch.matmul(emb, emb.t())
        n = sim.size(0)
        r, c = torch.triu_indices(n, n, offset=1, device=sim.device)
        div = 1.0 - sim[r, c]
        if div.numel(): scores.append(div.mean().item())
    return float(np.mean(scores)) if scores else 0.0


# ── Core evaluation (vectorized) ─────────────────────────────────────────────

def _evaluate_full(model, test_loader, top_k_list, metrics_list, device,
                   user_history, item_popularity=None, head_ratio=0.8, mid_ratio=0.1):
    # Collect ground truth
    if isinstance(test_loader.dataset, TensorDataset):
        all_users = test_loader.dataset.tensors[0].numpy()
        all_items = test_loader.dataset.tensors[1].numpy()
    else:
        us, it = [], []
        for u, i in test_loader:
            us.append(u.numpy()); it.append(i.numpy())
        all_users, all_items = np.concatenate(us), np.concatenate(it)

    user_gt = defaultdict(list)
    for u, i in zip(all_users, all_items):
        user_gt[int(u)].append(int(i))
    unique_users = np.array(sorted(user_gt.keys()))

    max_k   = max(top_k_list)
    n_items = model.n_items

    # Precompute log2 weights and IDCG lookup on device
    log2_w    = torch.tensor(_LOG2_RECIP[:max_k], dtype=torch.float32, device=device)
    idcg_lut  = torch.tensor(_IDCG_TABLE[:max_k + 1], dtype=torch.float32, device=device)

    # Group splits (Head, Mid, Tail)
    group_sets = {'Head': None, 'Mid': None, 'Tail': None}
    group_masks = {'Head': None, 'Mid': None, 'Tail': None}
    
    # Check if we need group-based metrics
    needed_groups = []
    for g in ['Head', 'Mid', 'Tail']:
        if any(m.startswith(g) for m in metrics_list) or (g == 'Tail' and 'LongTailCoverage' in metrics_list):
            needed_groups.append(g)
    # Support legacy LongTail metrics mapping to Tail
    if any(m.startswith('LongTail') for m in metrics_list):
        if 'Tail' not in needed_groups: needed_groups.append('Tail')

    if item_popularity is not None and needed_groups:
        head_set, mid_set, tail_set = get_item_split_sets(item_popularity, head_ratio, mid_ratio)
        group_sets['Head'], group_sets['Mid'], group_sets['Tail'] = head_set, mid_set, tail_set
        
        for g, s in group_sets.items():
            arr = np.zeros(n_items, dtype=bool)
            for idx in s:
                if idx < n_items: arr[idx] = True
            group_masks[g] = torch.tensor(arr, device=device)

    # Precompute popularity tensor for PopRatio
    pop_tensor = None
    if 'PopRatio' in metrics_list and item_popularity is not None:
        pop_tensor = torch.tensor(item_popularity, dtype=torch.float, device=device)
        mean_pop   = pop_tensor.mean().item()

    sums   = defaultdict(float)
    counts = defaultdict(int)
    all_top_k = []

    batch_size = test_loader.batch_size

    with torch.no_grad():
        for start in tqdm(range(0, len(unique_users), batch_size), desc="Eval", file=sys.stdout):
            u_ids = unique_users[start : start + batch_size]
            B     = len(u_ids)

            scores = model.forward(torch.tensor(u_ids, dtype=torch.long, device=device))  # (B, n_items)

            # Mask training items (keep GT items unmasked)
            mask_r, mask_c = [], []
            gt_lists = []
            for idx, u in enumerate(u_ids):
                gt  = user_gt[int(u)]
                gt_lists.append(gt)
                seen = user_history.get(int(u), set())
                excl = [it for it in seen if it not in set(gt)]
                if excl:
                    mask_r.extend([idx] * len(excl))
                    mask_c.extend(excl)
            if mask_r:
                scores[mask_r, mask_c] = -1e10

            _, top_idx = torch.topk(scores, k=max_k, dim=1)  # (B, max_k)
            all_top_k.extend(top_idx.cpu().tolist())

            # GT matrix (B, n_items) bool
            gt_mat = torch.zeros(B, n_items, dtype=torch.bool, device=device)
            for i, gts in enumerate(gt_lists):
                gt_mat[i, gts] = True

            gt_sizes = gt_mat.sum(dim=1).float()  # (B,)
            hit_mat = torch.gather(gt_mat.float(), 1, top_idx)  # (B, max_k)

            # Group-based hit matrices
            batch_group_data = {}
            for g in ['Head', 'Mid', 'Tail']:
                mask = group_masks[g]
                if mask is not None:
                    g_gt = gt_mat & mask
                    batch_group_data[g] = {
                        'hit': torch.gather(g_gt.float(), 1, top_idx),
                        'size': g_gt.sum(dim=1).float()
                    }

            for k in top_k_list:
                hk       = hit_mat[:, :k]
                hit_sum  = hk.sum(dim=1)
                n_rel    = gt_sizes.clamp(max=k).long()

                if 'HitRate'   in metrics_list: sums[f'HitRate@{k}']   += (hit_sum > 0).float().sum().item()
                if 'Recall'    in metrics_list: sums[f'Recall@{k}']    += (hit_sum / gt_sizes.clamp(min=1)).sum().item()
                if 'Precision' in metrics_list: sums[f'Precision@{k}'] += (hit_sum / k).sum().item()
                if 'NDCG'      in metrics_list:
                    dcg  = (hk * log2_w[:k]).sum(dim=1)
                    idcg = idcg_lut[n_rel]
                    sums[f'NDCG@{k}'] += (dcg / idcg.clamp(min=1e-8)).sum().item()

                # Group metrics
                for g in ['Head', 'Mid', 'Tail']:
                    if g in batch_group_data:
                        g_hit = batch_group_data[g]['hit'][:, :k]
                        g_sz = batch_group_data[g]['size']
                        g_mask = g_sz > 0
                        n_valid = g_mask.sum().item()
                        if n_valid == 0: continue
                        
                        h_v = g_hit[g_mask]; s_v = g_sz[g_mask]
                        h_sum = h_v.sum(dim=1)
                        
                        # Handle both current naming (HeadRecall) and legacy (LongTailRecall -> TailRecall)
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
        for m in ['HitRate', 'Recall', 'Precision', 'NDCG', 'PopRatio']:
            key = f'{m}@{k}'
            if key in sums: final[key] = sums[key] / n if n else 0.0
        
        for g in ['Head', 'Mid', 'Tail', 'LongTail']:
            for m in ['HitRate', 'Recall', 'NDCG']:
                key = f'{g}{m}@{k}'
                if key in sums: final[key] = sums[key] / counts.get(key, 1)

    return final, all_top_k, group_sets


def evaluate_metrics(model, data_loader, eval_config, device, test_loader, is_final=False):
    top_k_list  = eval_config.get('top_k', [10])
    metrics_list = eval_config.get('metrics', ['NDCG', 'Recall'])
    history     = data_loader.eval_user_history if is_final else data_loader.train_user_history
    item_pop    = data_loader.item_popularity
    
    # Ratios for Head/Mid/Tail
    head_ratio = eval_config.get('head_ratio', 0.8)
    mid_ratio = eval_config.get('mid_ratio', 0.1)

    final_results, all_top_k, group_sets = _evaluate_full(
        model, test_loader, top_k_list, metrics_list, device, history, item_pop, head_ratio, mid_ratio)

    # Embedding-based metrics
    emb = None
    with torch.no_grad():
        if hasattr(model, 'item_emb') and hasattr(model.item_emb, 'weight'):
            emb = model.item_emb.weight
        elif hasattr(model, 'item_embedding') and hasattr(model.item_embedding, 'weight'):
            emb = model.item_embedding.weight

    if 'GiniIndex_emb' in metrics_list and emb is not None:
        final_results['GiniIndex_emb'] = get_gini_index(emb.norm(dim=1).detach().cpu().numpy())

    for k in top_k_list:
        recs_k   = [sub[:k] for sub in all_top_k]
        flat_k   = [it for sub in recs_k for it in sub]

        if 'ILD' in metrics_list and emb is not None:
            final_results[f'ILD@{k}'] = get_ild(recs_k, emb)
        if 'Coverage' in metrics_list:
            final_results[f'Coverage@{k}'] = get_coverage(flat_k, data_loader.n_items)
        if 'GiniIndex' in metrics_list:
            final_results[f'GiniIndex@{k}'] = get_gini_index_from_recs(flat_k, data_loader.n_items)
        if 'Novelty' in metrics_list:
            final_results[f'Novelty@{k}'] = get_novelty(flat_k, item_pop)
        if 'Entropy' in metrics_list:
            final_results[f'Entropy@{k}'] = get_entropy_from_recs(flat_k)
            
        # Group Coverage
        for g in ['Head', 'Mid', 'Tail', 'LongTail']:
            m_name = f'{g}Coverage'
            if m_name in metrics_list:
                g_set = group_sets['Tail'] if g == 'LongTail' else group_sets[g]
                final_results[f'{m_name}@{k}'] = get_group_coverage(flat_k, g_set)

    return final_results
