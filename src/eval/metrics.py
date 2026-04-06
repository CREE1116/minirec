import torch
import numpy as np
from tqdm import tqdm
import sys
import pandas as pd
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader

# Precomputed 1/log2(rank+2) and cumulative IDCG table (up to K=10000)
_LOG2_RECIP  = (1.0 / np.log2(np.arange(2, 10002, dtype=np.float64))).astype(np.float32)
_IDCG_TABLE  = np.concatenate([[0.0], np.cumsum(_LOG2_RECIP)])  # index by n_relevant (0..10000)


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

def get_long_tail_item_set(item_popularity, head_volume_percent=0.8):
    pop = pd.Series(item_popularity) if isinstance(item_popularity, np.ndarray) else item_popularity
    sorted_pop = pop.sort_values(ascending=False)
    cutoff = sorted_pop.sum() * head_volume_percent
    head_n = int(np.searchsorted(sorted_pop.cumsum().values, cutoff, side='right'))
    return set(sorted_pop.index[head_n:].tolist())

def get_long_tail_coverage(flat_recs, item_popularity, head_volume_percent=0.8, precomputed_tail_set=None):
    tail = precomputed_tail_set or get_long_tail_item_set(item_popularity, head_volume_percent)
    if not tail: return 0.0
    return len(set(flat_recs) & tail) / len(tail)

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
                   user_history, item_popularity=None, long_tail_percent=0.8):
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
    log2_w    = torch.tensor(_LOG2_RECIP[:max_k], device=device)          # (max_k,)
    idcg_lut  = torch.tensor(_IDCG_TABLE[:max_k + 1], device=device)     # (max_k+1,)

    # Precompute tail mask
    tail_item_set = None
    tail_mask = None
    need_tail = any(m in metrics_list for m in [
        'LongTailHitRate', 'LongTailRecall', 'LongTailNDCG',
        'HeadHitRate', 'HeadRecall', 'HeadNDCG', 'LongTailCoverage'])
    if item_popularity is not None and need_tail:
        tail_item_set = get_long_tail_item_set(item_popularity, long_tail_percent)
        arr = np.zeros(n_items, dtype=bool)
        for idx in tail_item_set:
            if idx < n_items: arr[idx] = True
        tail_mask = torch.tensor(arr, device=device)  # (n_items,)

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

            # Hit matrix (B, max_k): 1 if top item is in GT
            hit_mat = torch.gather(gt_mat.float(), 1, top_idx)  # (B, max_k)

            # Tail/Head hit matrices (computed once per batch)
            if tail_mask is not None:
                tail_gt   = gt_mat & tail_mask          # (B, n_items)
                head_gt   = gt_mat & ~tail_mask
                tail_hit  = torch.gather(tail_gt.float(), 1, top_idx)
                head_hit  = torch.gather(head_gt.float(), 1, top_idx)
                tail_sizes = tail_gt.sum(dim=1).float()
                head_sizes = head_gt.sum(dim=1).float()

            for k in top_k_list:
                hk       = hit_mat[:, :k]          # (B, k)
                hit_sum  = hk.sum(dim=1)            # (B,)
                n_rel    = gt_sizes.clamp(max=k).long()

                if 'HitRate'   in metrics_list: sums[f'HitRate@{k}']   += (hit_sum > 0).float().sum().item()
                if 'Recall'    in metrics_list: sums[f'Recall@{k}']    += (hit_sum / gt_sizes.clamp(min=1)).sum().item()
                if 'Precision' in metrics_list: sums[f'Precision@{k}'] += (hit_sum / k).sum().item()
                if 'NDCG'      in metrics_list:
                    dcg  = (hk * log2_w[:k]).sum(dim=1)
                    idcg = idcg_lut[n_rel]
                    sums[f'NDCG@{k}'] += (dcg / idcg.clamp(min=1e-8)).sum().item()

                if tail_mask is not None:
                    tk = tail_hit[:, :k]; ts = tail_sizes; tm = ts > 0
                    hk2 = head_hit[:, :k]; hs = head_sizes; hm = hs > 0

                    for tag, hmat, sz, mask in [('LongTail', tk, ts, tm), ('Head', hk2, hs, hm)]:
                        n_valid = mask.sum().item()
                        if n_valid == 0: continue
                        h_v = hmat[mask]; s_v = sz[mask]
                        h_sum = h_v.sum(dim=1)
                        if f'{tag}HitRate'  in metrics_list:
                            sums[f'{tag}HitRate@{k}']  += (h_sum > 0).float().sum().item()
                            counts[f'{tag}HitRate@{k}'] += n_valid
                        if f'{tag}Recall'   in metrics_list:
                            sums[f'{tag}Recall@{k}']   += (h_sum / s_v.clamp(min=1)).sum().item()
                            counts[f'{tag}Recall@{k}'] += n_valid
                        if f'{tag}NDCG'     in metrics_list:
                            dcg_t  = (h_v * log2_w[:k]).sum(dim=1)
                            idcg_t = idcg_lut[s_v.clamp(max=k).long()]
                            sums[f'{tag}NDCG@{k}']   += (dcg_t / idcg_t.clamp(min=1e-8)).sum().item()
                            counts[f'{tag}NDCG@{k}'] += n_valid

                if pop_tensor is not None:
                    sums[f'PopRatio@{k}'] += (pop_tensor[top_idx[:, :k]].mean(dim=1) / mean_pop).sum().item()

                counts[k] += B

    n_total = len(unique_users)
    final = {}
    for k in top_k_list:
        n = counts[k]
        for m in ['HitRate', 'Recall', 'Precision', 'NDCG', 'PopRatio']:
            key = f'{m}@{k}'
            if key in sums: final[key] = sums[key] / n if n else 0.0
        for m in ['LongTailHitRate', 'LongTailRecall', 'LongTailNDCG',
                  'HeadHitRate',     'HeadRecall',     'HeadNDCG']:
            key = f'{m}@{k}'
            if key in sums: final[key] = sums[key] / counts.get(key, 1)

    return final, all_top_k, tail_item_set


def evaluate_metrics(model, data_loader, eval_config, device, test_loader, is_final=False):
    top_k_list  = eval_config.get('top_k', [10])
    metrics_list = eval_config.get('metrics', ['NDCG', 'Recall'])
    history     = data_loader.eval_user_history if is_final else data_loader.train_user_history
    item_pop    = data_loader.item_popularity
    lt_percent  = eval_config.get('long_tail_percent', 0.8)

    final_results, all_top_k, tail_set = _evaluate_full(
        model, test_loader, top_k_list, metrics_list, device, history, item_pop, lt_percent)

    # Embedding-based metrics (computed once, outside per-user loop)
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

        if 'ILD'             in metrics_list and emb is not None:
            final_results[f'ILD@{k}']             = get_ild(recs_k, emb)
        if 'Coverage'        in metrics_list:
            final_results[f'Coverage@{k}']        = get_coverage(flat_k, data_loader.n_items)
        if 'GiniIndex'       in metrics_list:
            final_results[f'GiniIndex@{k}']       = get_gini_index_from_recs(flat_k, data_loader.n_items)
        if 'LongTailCoverage' in metrics_list:
            final_results[f'LongTailCoverage@{k}'] = get_long_tail_coverage(flat_k, item_pop, lt_percent, tail_set)
        if 'Novelty'         in metrics_list:
            final_results[f'Novelty@{k}']         = get_novelty(flat_k, item_pop)
        if 'Entropy'         in metrics_list:
            final_results[f'Entropy@{k}']         = get_entropy_from_recs(flat_k)

    return final_results
