import torch
import numpy as np
from tqdm import tqdm
import sys
from collections import defaultdict

# Precomputed weights for NDCG on CPU (Numpy)
_MAX_K = 1000
_LOG2_RECIP = (1.0 / np.log2(np.arange(2, _MAX_K + 2))).astype(np.float32)
_IDCG_TABLE = np.cumsum(_LOG2_RECIP)

def calculate_gini(item_counts):
    """지니 계수 계산: 추천된 아이템 빈도의 불평등도 측정"""
    n = len(item_counts)
    if n == 0 or np.sum(item_counts) == 0: return 0.0
    sorted_counts = np.sort(item_counts)
    # Gini index formula: (2*sum(i*yi) / (n*sum(yi))) - (n+1)/n
    index = np.arange(1, n + 1)
    return (np.sum(2 * index * sorted_counts) / (n * np.sum(sorted_counts))) - (n + 1) / n

def _evaluate_full(model, test_loader, top_k_list, metrics_list, device, data_loader, is_final):
    gt_dict = data_loader.test_gt_dict if is_final else data_loader.valid_gt_dict
    history_dict = data_loader.eval_user_history if is_final else data_loader.train_user_history
    
    n_items = data_loader.n_items
    max_k = max(top_k_list)
    
    # 인기도 Split 준비
    item_popularity = getattr(data_loader, 'item_popularity', np.zeros(n_items))
    sorted_item_indices = np.argsort(item_popularity)[::-1]
    total_inter = np.sum(item_popularity)
    avg_pop_all = np.mean(item_popularity)
    
    head_ratio = data_loader.config.get('evaluation', {}).get('head_ratio', 0.8)
    mid_ratio = data_loader.config.get('evaluation', {}).get('mid_ratio', 0.1)
    
    head_thresh = total_inter * head_ratio
    mid_thresh = total_inter * (head_ratio + mid_ratio)
    
    cum_inter = np.cumsum(item_popularity[sorted_item_indices])
    head_items = set(sorted_item_indices[cum_inter <= head_thresh])
    mid_items = set(sorted_item_indices[(cum_inter > head_thresh) & (cum_inter <= mid_thresh)])
    tail_items = set(sorted_item_indices[cum_inter > mid_thresh])

    sums = defaultdict(float)
    counts = defaultdict(int) # 각 메트릭별 분모 (분할 메트릭용)
    all_recommended_items = {k: set() for k in top_k_list}
    item_rec_counts = {k: np.zeros(n_items) for k in top_k_list}
    total_valid_users = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Eval", file=sys.stdout):
            u_ids = batch[0].to(device)
            u_ids_np = u_ids.cpu().numpy()

            # 1. Inference & Masking
            scores = model.forward(u_ids)
            exclude_u, exclude_i = [], []
            for i, u in enumerate(u_ids_np):
                items = history_dict.get(int(u), [])
                exclude_u.extend([i] * len(items))
                exclude_i.extend(items)
            if exclude_u: scores[exclude_u, exclude_i] = -1e10

            # 2. Top-K extraction
            _, top_idx_gpu = torch.topk(scores, k=max_k, dim=1)
            top_idx = top_idx_gpu.cpu().numpy()
            
            for i, u in enumerate(u_ids_np):
                u_gt = gt_dict.get(int(u), set())
                if not u_gt: continue
                
                total_valid_users += 1
                u_top_k = top_idx[i]
                
                # 정답지 분할
                u_gt_head = u_gt & head_items
                u_gt_mid = u_gt & mid_items
                u_gt_tail = u_gt & tail_items

                for k in top_k_list:
                    current_top_k = u_top_k[:k]
                    all_recommended_items[k].update(current_top_k)
                    for it in current_top_k: item_rec_counts[k][it] += 1
                    
                    hits_k = [1 if it in u_gt else 0 for it in current_top_k]
                    hit_count = sum(hits_k)
                    
                    # ── Basic Metrics ──────────────────────────────────────
                    sums[f'Recall@{k}'] += hit_count / len(u_gt)
                    sums[f'NDCG@{k}'] += sum([h * _LOG2_RECIP[r] for r, h in enumerate(hits_k)]) / _IDCG_TABLE[min(k, len(u_gt)) - 1]
                    sums[f'HitRate@{k}'] += 1 if hit_count > 0 else 0
                    sums[f'Precision@{k}'] += hit_count / k
                    sums[f'uRecall@{k}'] += hit_count / min(k, len(u_gt))
                    sums[f'uNDCG@{k}'] += sum([h * _LOG2_RECIP[r] for r, h in enumerate(hits_k)]) / _IDCG_TABLE[min(k, len(u_gt)) - 1]

                    # ── Popularity Splits ──────────────────────────
                    # Head
                    if u_gt_head:
                        h_hits = [1 if it in u_gt_head else 0 for it in current_top_k]
                        h_hit_cnt = sum(h_hits)
                        sums[f'HeadRecall@{k}'] += h_hit_cnt / len(u_gt_head)
                        sums[f'HeadNDCG@{k}'] += sum([h * _LOG2_RECIP[r] for r, h in enumerate(h_hits)]) / _IDCG_TABLE[min(k, len(u_gt_head)) - 1]
                        sums[f'HeadHitRate@{k}'] += 1 if h_hit_cnt > 0 else 0
                        counts[f'HeadRecall@{k}'] += 1
                    
                    # Mid
                    if u_gt_mid:
                        m_hits = [1 if it in u_gt_mid else 0 for it in current_top_k]
                        m_hit_cnt = sum(m_hits)
                        sums[f'MidRecall@{k}'] += m_hit_cnt / len(u_gt_mid)
                        sums[f'MidNDCG@{k}'] += sum([h * _LOG2_RECIP[r] for r, h in enumerate(m_hits)]) / _IDCG_TABLE[min(k, len(u_gt_mid)) - 1]
                        sums[f'MidHitRate@{k}'] += 1 if m_hit_cnt > 0 else 0
                        counts[f'MidRecall@{k}'] += 1

                    # Tail
                    if u_gt_tail:
                        t_hits = [1 if it in u_gt_tail else 0 for it in current_top_k]
                        t_hit_cnt = sum(t_hits)
                        sums[f'TailRecall@{k}'] += t_hit_cnt / len(u_gt_tail)
                        sums[f'TailNDCG@{k}'] += sum([h * _LOG2_RECIP[r] for r, h in enumerate(t_hits)]) / _IDCG_TABLE[min(k, len(u_gt_tail)) - 1]
                        sums[f'TailHitRate@{k}'] += 1 if t_hit_cnt > 0 else 0
                        counts[f'TailRecall@{k}'] += 1
                        
                    # ── Beyond-Accuracy ──────────────────────────────────
                    u_novelty = sum([-np.log2((item_popularity[it]+1)/total_inter) for it in current_top_k])
                    sums[f'Novelty@{k}'] += u_novelty / k
                    u_pop = np.mean([item_popularity[it] for it in current_top_k])
                    sums[f'PopRatio@{k}'] += u_pop / avg_pop_all

    # 6. Final Average
    final = {}
    n = total_valid_users + 1e-12
    for k in top_k_list:
        for m in metrics_list:
            key = f'{m}@{k}'
            # 분할 메트릭은 해당 카운트로 나눔, 기본 메트릭은 n으로 나눔
            if key in sums:
                denom = counts.get(key, n) if 'Recall' in m or 'NDCG' in m or 'HitRate' in m else n
                if 'Head' in m or 'Mid' in m or 'Tail' in m: denom = counts.get(m.replace('HitRate', 'Recall').replace('NDCG', 'Recall') + f'@{k}', n)
                final[key] = float(sums[key] / denom)
            
        # Catalog-based Metrics
        final[f'Coverage@{k}'] = len(all_recommended_items[k]) / n_items
        final[f'HeadCoverage@{k}'] = len(all_recommended_items[k] & head_items) / len(head_items) if head_items else 0
        final[f'MidCoverage@{k}'] = len(all_recommended_items[k] & mid_items) / len(mid_items) if mid_items else 0
        final[f'TailCoverage@{k}'] = len(all_recommended_items[k] & tail_items) / len(tail_items) if tail_items else 0
        final[f'GiniIndex@{k}'] = calculate_gini(item_rec_counts[k])
            
    return final

def evaluate_metrics(model, data_loader, eval_config, device, test_loader, is_final=False):
    top_k = eval_config.get('top_k', [10, 20, 50, 100])
    metrics = eval_config.get('metrics', ["Recall", "NDCG", "HitRate"])
    return _evaluate_full(model, test_loader, top_k, metrics, device, data_loader, is_final)
