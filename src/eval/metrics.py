import torch
import numpy as np
from tqdm import tqdm
import sys
import pandas as pd
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader

# Precomputed weights
_LOG2_RECIP  = (1.0 / np.log2(np.arange(2, 10002, dtype=np.float64))).astype(np.float32)
_IDCG_TABLE  = np.concatenate([[np.float32(0.0)], np.cumsum(_LOG2_RECIP)])


def get_gini_index_gpu(counts_tensor):
    if counts_tensor.sum() == 0: return 0.0
    n = counts_tensor.size(0)
    values, _ = torch.sort(counts_tensor)
    idx = torch.arange(1, n + 1, device=counts_tensor.device, dtype=torch.float32)
    gini = torch.sum((2 * idx - n - 1) * values) / (n * values.sum() + 1e-12)
    return gini.item()

def get_novelty_gpu(counts_tensor, item_popularity_tensor, num_users):
    if counts_tensor.sum() == 0: return 0.0
    total_train = item_popularity_tensor.sum()
    p_i = (item_popularity_tensor + 1) / (total_train + item_popularity_tensor.size(0))
    self_info = -torch.log2(p_i + 1e-10)
    return (torch.sum(counts_tensor * self_info) / (counts_tensor.sum() + 1e-12)).item()

def get_item_split_sets(item_popularity, head_ratio=0.8, mid_ratio=0.1):
    pop = pd.Series(item_popularity)
    sorted_pop = pop.sort_values(ascending=False)
    total_vol = sorted_pop.sum()
    cumsum_vol = sorted_pop.cumsum().values
    h_n = int(np.searchsorted(cumsum_vol, total_vol * head_ratio, side='right'))
    m_n = int(np.searchsorted(cumsum_vol, total_vol * (head_ratio + mid_ratio), side='right'))
    return sorted_pop.index[:h_n].tolist(), sorted_pop.index[h_n:m_n].tolist(), sorted_pop.index[m_n:].tolist()


def _evaluate_full(model, test_loader, top_k_list, metrics_list, device,
                   user_history_sp, test_gt_sp, item_popularity=None, head_ratio=0.8, mid_ratio=0.1):
    n_users, n_items = test_gt_sp.shape
    max_k = max(top_k_list)
    log2_w = torch.tensor(_LOG2_RECIP[:max_k], dtype=torch.float32, device=device)
    idcg_lut = torch.tensor(_IDCG_TABLE[:max_k + 1], dtype=torch.float32, device=device)

    # 1. Pre-transfer static data to GPU
    pop_tensor = None; mean_pop = 1.0
    if item_popularity is not None:
        pop_tensor = torch.tensor(item_popularity, dtype=torch.float32, device=device)
        mean_pop = pop_tensor.mean().item()

    group_masks = {}
    needed_groups = []
    if any('Head' in m for m in metrics_list): needed_groups.append('Head')
    if any('Mid' in m for m in metrics_list): needed_groups.append('Mid')
    if any('Tail' in m for m in metrics_list) or any('LongTail' in m for m in metrics_list): 
        needed_groups.append('Tail')
    
    if item_popularity is not None and needed_groups:
        h_idx, m_idx, t_idx = get_item_split_sets(item_popularity, head_ratio, mid_ratio)
        group_map = {'Head': h_idx, 'Mid': m_idx, 'Tail': t_idx}
        for g in needed_groups:
            mask = torch.zeros(n_items, dtype=torch.bool, device=device)
            idxs = group_map[g]
            if idxs: mask[torch.tensor(idxs, dtype=torch.long, device=device)] = True
            group_masks[g] = mask

    inv_pscore = None
    if any(m.startswith('u') for m in metrics_list) and item_popularity is not None:
        pscore = torch.pow((pop_tensor + 1) / (pop_tensor.max() + 1), 0.5)
        inv_pscore = 1.0 / pscore

    sums = defaultdict(float); counts = defaultdict(float)
    rec_counts_dict = {k: torch.zeros(n_items, device=device, dtype=torch.float32) for k in top_k_list}
    history_coo = user_history_sp.coalesce(); gt_coo = test_gt_sp.coalesce()

    # 2. Evaluation Loop
    batch_size = test_loader.batch_size

    with torch.no_grad():
        for start in tqdm(range(0, n_users, batch_size), desc="Eval (GPU)", file=sys.stdout):
            u_ids = torch.arange(start, min(start + batch_size, n_users), device=device)
            B = u_ids.size(0)

            # Prediction
            scores = model.forward(u_ids)

            # Masking
            batch_hist = torch.index_select(history_coo, 0, u_ids).coalesce()
            if batch_hist.indices().size(1) > 0:
                scores[batch_hist.indices()[0], batch_hist.indices()[1]] = -1e10

            # Top-K
            _, top_idx = torch.topk(scores, k=max_k, dim=1)

            # GT Processing
            batch_gt_sp = torch.index_select(gt_coo, 0, u_ids).coalesce()
            gt_dense_batch = torch.zeros((B, n_items), dtype=torch.bool, device=device)
            if batch_gt_sp.indices().size(1) > 0:
                gt_dense_batch[batch_gt_sp.indices()[0], batch_gt_sp.indices()[1]] = True

            gt_sizes = gt_dense_batch.sum(dim=1).float()
            hit_mat = torch.gather(gt_dense_batch.float(), 1, top_idx)

            # Unbiased vectorization
            if inv_pscore is not None:
                batch_gt_weights = gt_dense_batch.float() * inv_pscore.unsqueeze(0)
                sorted_gt_weights, _ = torch.sort(batch_gt_weights, dim=1, descending=True)
                u_gt_sum = sorted_gt_weights.sum(dim=1)
                u_idcg_matrix = sorted_gt_weights[:, :max_k] * log2_w
                rank_inv_pscore = inv_pscore[top_idx]

            # Accumulate Metrics
            for k in top_k_list:
                rec_counts_dict[k].scatter_add_(0, top_idx[:, :k].flatten(), torch.ones(B * k, device=device))
                hk = hit_mat[:, :k]; hit_sum = hk.sum(dim=1)

                if 'HitRate' in metrics_list: sums[f'HitRate@{k}'] += (hit_sum > 0).float().sum().item()
                if 'Recall' in metrics_list: sums[f'Recall@{k}'] += (hit_sum / gt_sizes.clamp(min=1)).sum().item()
                if 'Precision' in metrics_list: sums[f'Precision@{k}'] += (hit_sum / k).sum().item()
                if 'NDCG' in metrics_list:
                    dcg = (hk * log2_w[:k]).sum(dim=1)
                    sums[f'NDCG@{k}'] += (dcg / idcg_lut[gt_sizes.clamp(max=k).long()].clamp(min=1e-8)).sum().item()

                if inv_pscore is not None:
                    u_hk = hk * rank_inv_pscore[:, :k]
                    if 'uRecall' in metrics_list: sums[f'uRecall@{k}'] += (u_hk.sum(dim=1) / u_gt_sum.clamp(min=1e-8)).sum().item()
                    if 'uNDCG' in metrics_list: sums[f'uNDCG@{k}'] += (u_hk.sum(dim=1) / u_idcg_matrix[:, :k].sum(dim=1).clamp(min=1e-8)).sum().item()

                for g in needed_groups:
                    mask = group_masks[g]; g_gt = gt_dense_batch & mask
                    g_sz = g_gt.sum(dim=1).float(); v_mask = g_sz > 0; n_v = v_mask.sum().item()
                    if n_v > 0:
                        h_v_all = torch.gather(g_gt[v_mask].float(), 1, top_idx[v_mask, :k])
                        h_sum = h_v_all.sum(dim=1)
                        s_v = g_sz[v_mask]
                        
                        tags = [g, 'LongTail'] if g == 'Tail' else [g]
                        for tag in tags:
                            if f'{tag}HitRate' in metrics_list: sums[f'{tag}HitRate@{k}'] += (h_sum > 0).float().sum().item(); counts[f'{tag}HitRate@{k}'] += n_v
                            if f'{tag}Recall' in metrics_list: sums[f'{tag}Recall@{k}'] += (h_sum / s_v).sum().item(); counts[f'{tag}Recall@{k}'] += n_v
                            if f'{tag}NDCG' in metrics_list:
                                dcg_v = (h_v_all * log2_w[:k]).sum(dim=1)
                                sums[f'{tag}NDCG@{k}'] += (dcg_v / idcg_lut[s_v.clamp(max=k).long()].clamp(min=1e-8)).sum().item(); counts[f'{tag}NDCG@{k}'] += n_v

                if 'PopRatio' in metrics_list:
                    sums[f'PopRatio@{k}'] += (pop_tensor[top_idx[:, :k]].mean(dim=1) / mean_pop).sum().item()
                counts[k] += B

    final = {}
    for k in top_k_list:
        # Denominator is the number of users actually evaluated
        n = counts[k]
        for m in ['HitRate', 'Recall', 'uRecall', 'Precision', 'NDCG', 'uNDCG', 'PopRatio']:
            key = f'{m}@{k}'; final[key] = sums[key] / n if n else 0.0
        for g in ['Head', 'Mid', 'Tail', 'LongTail']:
            for m in ['HitRate', 'Recall', 'NDCG']:
                key = f'{g}{m}@{k}'; final[key] = sums[key] / (counts.get(key, 0) + 1e-12)
    return final, rec_counts_dict, group_masks


def evaluate_metrics(model, data_loader, eval_config, device, test_loader, is_final=False):
    history_sp = data_loader.sp_train_valid if is_final else data_loader.sp_train
    test_gt_sp = data_loader.sp_test_gt if is_final else data_loader.sp_valid_gt

    final_res, rec_counts, g_masks = _evaluate_full(
        model, test_loader, 
        top_k_list=eval_config.get('top_k', [10]), 
        metrics_list=eval_config.get('metrics', []), 
        device=device, 
        user_history_sp=history_sp.to(device), 
        test_gt_sp=test_gt_sp.to(device), 
        item_popularity=data_loader.item_popularity,
        head_ratio=eval_config.get('head_ratio', 0.8),
        mid_ratio=eval_config.get('mid_ratio', 0.1)
    )

    if data_loader.item_popularity is not None:
        pop_t = torch.tensor(data_loader.item_popularity, dtype=torch.float32, device=device)
        for k in eval_config.get('top_k', [10]):
            rc = rec_counts[k]
            if 'Coverage' in eval_config['metrics']: final_res[f'Coverage@{k}'] = (rc > 0).float().mean().item()
            if 'GiniIndex' in eval_config['metrics']: final_res[f'GiniIndex@{k}'] = get_gini_index_gpu(rc)
            if 'Novelty' in eval_config['metrics']: final_res[f'Novelty@{k}'] = get_novelty_gpu(rc, pop_t, data_loader.n_users)
            for g in ['Head', 'Mid', 'Tail', 'LongTail']:
                if f'{g}Coverage' in eval_config['metrics']:
                    # Use Tail mask for LongTail
                    mask_key = 'Tail' if g == 'LongTail' else g
                    mask = g_masks.get(mask_key)
                    if mask is not None: final_res[f'{g}Coverage@{k}'] = (rc[mask] > 0).float().mean().item()
    return final_res
