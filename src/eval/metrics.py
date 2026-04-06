import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
import json
from scipy.stats import gmean
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from collections import defaultdict
import sys

# Precomputed 1/log2(rank+2) table for NDCG (covers K up to 10000).
_LOG2_RECIP = (1.0 / np.log2(np.arange(2, 10002, dtype=np.float64))).astype(np.float32)

def get_hit_rate(pred_list, ground_truth):
    intersection = set(pred_list).intersection(set(ground_truth))
    return 1 if len(intersection) > 0 else 0

def get_recall(pred_list, ground_truth):
    if len(ground_truth) == 0: return 0.0
    intersection = set(pred_list).intersection(set(ground_truth))
    return len(intersection) / len(ground_truth)

def get_precision(pred_list, ground_truth):
    if len(pred_list) == 0: return 0.0
    intersection = set(pred_list).intersection(set(ground_truth))
    return len(intersection) / len(pred_list)

def get_ndcg(pred_list, ground_truth):
    if not ground_truth: return 0.0
    gt_set = ground_truth if isinstance(ground_truth, (set, frozenset)) else set(ground_truth)
    k = len(pred_list)
    hits = np.array([1 if item in gt_set else 0 for item in pred_list], dtype=np.float32)
    dcg  = float(np.dot(hits, _LOG2_RECIP[:k]))
    n_relevant = min(len(gt_set), k)
    idcg = float(_LOG2_RECIP[:n_relevant].sum())
    return dcg / idcg if idcg > 0 else 0.0

def get_pop_ratio(target_item, item_popularity, mean_pop):
    if hasattr(item_popularity, 'get'): target_pop = item_popularity.get(target_item, 1)
    else:
        try: target_pop = item_popularity[target_item]
        except (IndexError, KeyError): target_pop = 1
    return (target_pop / mean_pop) if mean_pop > 0 else 1.0

def get_coverage(all_recommended_items, n_items):
    if n_items == 0: return 0.0
    return len(set(all_recommended_items)) / n_items

def get_gini_index(values):
    if len(values) == 0: return 0.0
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    if np.sum(values) == 0: return 0.0
    return (np.sum((2 * index - n - 1) * values)) / (n * np.sum(values))

def get_gini_index_from_recs(recommended_items, num_items):
    if not recommended_items: return 0.0
    item_counts = np.zeros(num_items)
    for item in recommended_items:
        if item < num_items: item_counts[item] += 1
    return get_gini_index(item_counts)

def get_novelty(recommended_items, item_popularity, num_users=None):
    if not recommended_items: return 0.0
    recommended_items = np.array(recommended_items)
    if isinstance(item_popularity, pd.Series): pop_values = item_popularity.values
    elif isinstance(item_popularity, dict):
        all_ids = list(item_popularity.keys())
        if recommended_items.size > 0: all_ids.append(recommended_items.max())
        max_id = max(all_ids)
        pop_values = np.zeros(max_id + 1)
        for k, v in item_popularity.items(): pop_values[k] = v
    else: pop_values = np.array(item_popularity)
    
    total = pop_values.sum() if num_users is None else num_users
    n_items = len(pop_values)
    counts = pop_values[recommended_items]
    p_i = (counts + 1) / (total + n_items)
    return np.mean(-np.log2(p_i + 1e-10))

def get_long_tail_item_set(item_popularity, head_volume_percent=0.8):
    if isinstance(item_popularity, np.ndarray): item_popularity = pd.Series(item_popularity)
    sorted_pop = item_popularity.sort_values(ascending=False)
    cumsum = sorted_pop.cumsum()
    cutoff = sorted_pop.sum() * head_volume_percent
    head_indices = np.searchsorted(cumsum.values, cutoff, side='right')
    return set(sorted_pop.index[head_indices:].tolist())

def get_long_tail_coverage(all_recommended_items, item_popularity, head_volume_percent=0.8, precomputed_tail_set=None):
    if not all_recommended_items: return 0.0
    long_tail_item_ids = precomputed_tail_set if precomputed_tail_set is not None else get_long_tail_item_set(item_popularity, head_volume_percent)
    if len(long_tail_item_ids) == 0: return 0.0
    intersection = set(all_recommended_items).intersection(long_tail_item_ids)
    return len(intersection) / len(long_tail_item_ids)

def get_entropy_from_recs(recommended_items):
    if not recommended_items: return 0.0
    counts = {}
    for item in recommended_items: counts[item] = counts.get(item, 0) + 1
    popularity = np.array(list(counts.values()))
    if len(popularity) == 0: return 0.0
    probs = popularity / np.sum(popularity)
    return -np.sum(probs * np.log2(probs))

def get_gini_index_emb(norms):
    return get_gini_index(norms)

def get_ild(all_top_k_items, item_embeddings):
    if not all_top_k_items: return 0.0
    scores = []
    for user_recs in all_top_k_items:
        if len(user_recs) < 2: continue
        rec_embeds = item_embeddings[user_recs]
        norms = rec_embeds.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        rec_embeds_norm = rec_embeds / norms
        sim_matrix = torch.matmul(rec_embeds_norm, rec_embeds_norm.transpose(0, 1))
        sim_matrix = torch.nan_to_num(sim_matrix, nan=1.0)
        n = sim_matrix.size(0)
        row_idx, col_idx = torch.triu_indices(n, n, offset=1, device=sim_matrix.device)
        pairwise_divs = 1.0 - sim_matrix[row_idx, col_idx]
        if pairwise_divs.numel() > 0: scores.append(pairwise_divs.mean().item())
    return float(np.mean(scores)) if scores else 0.0

def _evaluate_full(model, test_loader, top_k_list, metrics_list, device, user_history, item_popularity=None, long_tail_percent=0.8):
    if isinstance(test_loader.dataset, TensorDataset):
        all_users_np = test_loader.dataset.tensors[0].numpy()
        all_items_np = test_loader.dataset.tensors[1].numpy()
    else:
        all_users, all_items = [], []
        for u_batch, i_batch in test_loader:
            all_users.append(u_batch.numpy())
            all_items.append(i_batch.numpy())
        all_users_np, all_items_np = np.concatenate(all_users), np.concatenate(all_items)

    user_gt = defaultdict(list)
    for u, i in zip(all_users_np, all_items_np): user_gt[u].append(i)
    unique_users = sorted(list(user_gt.keys()))
    
    # metrics_list에서 _evaluate_full이 직접 계산하는 것들만 results에 포함
    core_metrics = ['HitRate', 'Recall', 'Precision', 'NDCG', 'LongTailHitRate', 'LongTailNDCG', 'HeadHitRate', 'HeadNDCG']
    active_core_metrics = [m for m in metrics_list if m in core_metrics]
    
    results = {f'{m}@{k}': [] for k in top_k_list for m in active_core_metrics}
    all_top_k = []
    
    pop_ratio_raw = {k: [] for k in top_k_list}
    mean_pop = np.mean(item_popularity) if item_popularity is not None else None
    tail_item_set = get_long_tail_item_set(item_popularity, long_tail_percent) if item_popularity is not None else None

    batch_size = test_loader.batch_size
    max_k = max(top_k_list) if top_k_list else 10
    
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_users), batch_size), desc="Eval (Full)", file=sys.stdout):
            u_ids = unique_users[i:i+batch_size]
            u_tensor = torch.LongTensor(u_ids).to(device)
            scores = model.forward(u_tensor)
            
            mask_rows, mask_cols = [], []
            for idx, u_id in enumerate(u_ids):
                seen = user_history.get(u_id, set())
                gt = set(user_gt[u_id])
                # seen 아이템 중 gt가 아닌 것만 마스킹 대상으로 지정
                exclude = [it for it in seen if it not in gt]
                mask_rows.extend([idx] * len(exclude))
                mask_cols.extend(exclude)
            
            if mask_rows:
                # 마스킹: 학습에 사용된 아이템들을 최하위 점수로 밀어냄
                scores[mask_rows, mask_cols] = -1e10

            _, top_indices = torch.topk(scores, k=max_k, dim=1)
            top_indices = top_indices.cpu().numpy()
            
            for idx, u_id in enumerate(u_ids):
                pred_list = top_indices[idx].tolist()
                gt = user_gt[u_id]
                all_top_k.append(pred_list)
                
                for k in top_k_list:
                    pk = pred_list[:k]
                    if 'HitRate' in metrics_list: results[f'HitRate@{k}'].append(get_hit_rate(pk, gt))
                    if 'Recall' in metrics_list: results[f'Recall@{k}'].append(get_recall(pk, gt))
                    if 'Precision' in metrics_list: results[f'Precision@{k}'].append(get_precision(pk, gt))
                    if 'NDCG' in metrics_list: results[f'NDCG@{k}'].append(get_ndcg(pk, gt))
                    
                    if tail_item_set is not None:
                        tail_gt = [it for it in gt if it in tail_item_set]
                        if tail_gt:
                            if 'LongTailHitRate' in metrics_list: results[f'LongTailHitRate@{k}'].append(get_hit_rate(pk, tail_gt))
                            if 'LongTailNDCG' in metrics_list: results[f'LongTailNDCG@{k}'].append(get_ndcg(pk, tail_gt))
                        head_gt = [it for it in gt if it not in tail_item_set]
                        if head_gt:
                            if 'HeadHitRate' in metrics_list: results[f'HeadHitRate@{k}'].append(get_hit_rate(pk, head_gt))
                            if 'HeadNDCG' in metrics_list: results[f'HeadNDCG@{k}'].append(get_ndcg(pk, head_gt))
                    
                    if 'PopRatio' in metrics_list and mean_pop is not None:
                        # 추천 아이템 전체 대상으로 계산 (hit 여부 무관)
                        pop_ratio_raw[k].append(np.mean([get_pop_ratio(it, item_popularity, mean_pop) for it in pk]))

    final = {k: np.nanmean(v) if v else 0.0 for k, v in results.items()}
    if 'PopRatio' in metrics_list:
        for k in top_k_list: final[f'PopRatio@{k}'] = np.mean(pop_ratio_raw[k]) if pop_ratio_raw[k] else 0.0
    return final, all_top_k, tail_item_set

def evaluate_metrics(model, data_loader, eval_config, device, test_loader, is_final=False):
    top_k_list = eval_config.get('top_k', [10])
    metrics_list = eval_config.get('metrics', ['NDCG', 'Recall'])
    history = data_loader.eval_user_history if is_final else data_loader.train_user_history
    item_pop = data_loader.item_popularity
    lt_percent = eval_config.get('long_tail_percent', 0.8)
    
    final_results, all_top_k, precomputed_tail_set = _evaluate_full(model, test_loader, top_k_list, metrics_list, device, history, item_pop, lt_percent)

    # Global/Diversity Metrics
    emb_for_ild = None
    item_norms = None
    if any(m in metrics_list for m in ['ILD', 'GiniIndex_emb']):
        with torch.no_grad():
            # MF 모델 등에서 임베딩 가중치 가져오기
            if hasattr(model, 'item_emb') and hasattr(model.item_emb, 'weight'):
                emb_for_ild = model.item_emb.weight
            elif hasattr(model, 'item_embedding') and hasattr(model.item_embedding, 'weight'):
                emb_for_ild = model.item_embedding.weight

            if 'GiniIndex_emb' in metrics_list and emb_for_ild is not None:
                item_norms = torch.norm(emb_for_ild, dim=1).detach().cpu().numpy()

    if 'GiniIndex_emb' in metrics_list and item_norms is not None:
        final_results['GiniIndex_emb'] = get_gini_index_emb(item_norms)

    for k in top_k_list:
        recs_at_k = [sub[:k] for sub in all_top_k]
        flat_at_k = [it for sub in recs_at_k for it in sub]

        if 'ILD' in metrics_list and emb_for_ild is not None:
            final_results[f'ILD@{k}'] = get_ild(recs_at_k, emb_for_ild)
        if 'Coverage' in metrics_list:
            final_results[f'Coverage@{k}'] = get_coverage(flat_at_k, data_loader.n_items)
        if 'GiniIndex' in metrics_list:
            final_results[f'GiniIndex@{k}'] = get_gini_index_from_recs(flat_at_k, data_loader.n_items)
        if 'LongTailCoverage' in metrics_list:
            # _evaluate_full에서 이미 계산한 tail_set 재사용
            final_results[f'LongTailCoverage@{k}'] = get_long_tail_coverage(flat_at_k, item_pop, lt_percent, precomputed_tail_set)
        if 'Novelty' in metrics_list:
            final_results[f'Novelty@{k}'] = get_novelty(flat_at_k, item_pop)
        if 'Entropy' in metrics_list:
            final_results[f'Entropy@{k}'] = get_entropy_from_recs(flat_at_k)

    return final_results
