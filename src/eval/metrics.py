import torch
import numpy as np
from tqdm import tqdm
import sys
from collections import defaultdict

# Precomputed weights for NDCG
_LOG2 = np.log2(np.arange(2, 10002, dtype=np.float64))
_LOG2_RECIP = (1.0 / _LOG2).astype(np.float32)

def get_gini_index(counts):
    """CPU-based efficient Gini Index"""
    if counts.sum() == 0: return 0.0
    n = len(counts)
    values = np.sort(counts)
    idx = np.arange(1, n + 1)
    gini = np.sum((2 * idx - n - 1) * values) / (n * values.sum() + 1e-12)
    return float(gini)

def _evaluate_full(model, test_loader, top_k_list, metrics_list, device,
                   user_history_sp, test_gt_sp, item_popularity=None):
    n_users, n_items = test_gt_sp.shape
    max_k = max(top_k_list)
    
    # 1. Pre-process Ground Truth into Sets for O(1) lookup
    # This is done once per evaluation call.
    gt_coo = test_gt_sp.to("cpu").coalesce()
    gt_indices = gt_coo.indices().numpy()
    gt_dict = defaultdict(set)
    for u, i in zip(gt_indices[0], gt_indices[1]):
        gt_dict[u].add(i)
    
    # 2. Pre-process User History for masking on GPU
    hist_coo = user_history_sp.coalesce().to(device)
    hist_indices = hist_coo.indices()
    
    # 3. Item Grouping (Optional)
    needed_groups = [g for g in ['Head', 'Mid', 'Tail'] if any(g in m for m in metrics_list)]
    item_to_group = None
    if item_popularity is not None and needed_groups:
        # Simple top-20% as head
        sorted_items = np.argsort(item_popularity)[::-1]
        head_idx = set(sorted_items[:int(n_items * 0.2)])
        item_to_group = np.full(n_items, 2) # Default as Tail (2)
        for idx in head_idx: item_to_group[idx] = 0 # Head as 0
        # Mid omitted for simplicity or can be added similarly

    sums = defaultdict(float)
    counts = defaultdict(float)
    all_rec_counts = {k: np.zeros(n_items) for k in top_k_list}

    batch_size = test_loader.batch_size
    with torch.no_grad():
        for start in tqdm(range(0, n_users, batch_size), desc="Eval (Optimized)", file=sys.stdout):
            end = min(start + batch_size, n_users)
            u_ids = torch.arange(start, end, device=device)
            B = end - start

            # [GPU] 1. Inference
            scores = model.forward(u_ids) # (B, I)

            # [GPU] 2. Efficient Masking
            # Filter global history indices to current batch
            mask = (hist_indices[0] >= start) & (hist_indices[0] < end)
            if mask.any():
                scores[hist_indices[0, mask] - start, hist_indices[1, mask]] = -1e10

            # [GPU] 3. Top-K extraction
            _, top_idx_gpu = torch.topk(scores, k=max_k, dim=1)
            
            # [Transfer] Move only result to CPU
            top_idx = top_idx_gpu.cpu().numpy() # (B, max_k)
            
            # [CPU] 4. Metric Calculation (Fast loop with Sets)
            for i in range(B):
                u_global = start + i
                u_gt = gt_dict.get(u_global, set())
                if not u_gt:
                    continue
                
                # Pre-calculate hits for all K
                hits = np.array([1 if item in u_gt else 0 for item in top_idx[i]])
                gt_size = len(u_gt)
                
                for k in top_k_list:
                    k_hits = hits[:k]
                    hit_count = np.sum(k_hits)
                    
                    if 'Recall' in metrics_list:
                        sums[f'Recall@{k}'] += hit_count / min(k, gt_size)
                    if 'HitRate' in metrics_list:
                        sums[f'HitRate@{k}'] += 1 if hit_count > 0 else 0
                    if 'NDCG' in metrics_list:
                        dcg = np.sum(k_hits * _LOG2_RECIP[:k])
                        idcg = np.sum(_LOG2_RECIP[:min(k, gt_size)])
                        sums[f'NDCG@{k}'] += dcg / idcg
                    
                    # Recommendation counts for Coverage/Gini
                    all_rec_counts[k][top_idx[i, :k]] += 1
                
                counts['total'] += 1 # Only count users with GT

    # Final Average
    final = {}
    n = counts['total'] + 1e-12
    for k in top_k_list:
        for m in ['HitRate', 'Recall', 'NDCG']:
            key = f'{m}@{k}'
            final[key] = sums[key] / n
        
        if 'Coverage' in metrics_list:
            final[f'Coverage@{k}'] = np.mean(all_rec_counts[k] > 0)
        if 'GiniIndex' in metrics_list:
            final[f'GiniIndex@{k}'] = get_gini_index(all_rec_counts[k])
            
    return final

def evaluate_metrics(model, data_loader, eval_config, device, test_loader, is_final=False):
    history_sp = data_loader.sp_train_valid if is_final else data_loader.sp_train
    test_gt_sp = data_loader.sp_test_gt if is_final else data_loader.sp_valid_gt

    metrics = evaluate_metrics_optimized(
        model, test_loader, 
        top_k_list=eval_config.get('top_k', [10, 20]), 
        metrics_list=eval_config.get('metrics', ['Recall', 'NDCG']), 
        device=device, 
        user_history_sp=history_sp, 
        test_gt_sp=test_gt_sp, 
        item_popularity=data_loader.item_popularity
    )
    return metrics

def evaluate_metrics_optimized(model, test_loader, top_k_list, metrics_list, device,
                               user_history_sp, test_gt_sp, item_popularity=None):
    # This is the entry point called by Trainer
    return _evaluate_full(model, test_loader, top_k_list, metrics_list, device, 
                          user_history_sp, test_gt_sp, item_popularity)
