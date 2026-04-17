import torch
import numpy as np
from tqdm import tqdm
import sys
from collections import defaultdict

# Precomputed weights for NDCG
_LOG2 = np.log2(np.arange(2, 10002, dtype=np.float64))
_LOG2_RECIP = (1.0 / _LOG2).astype(np.float32)

def _evaluate_full(model, test_loader, top_k_list, metrics_list, device,
                   user_history_sp, test_gt_sp, item_popularity=None):
    n_users, n_items = test_gt_sp.shape
    max_k = max(top_k_list)
    
    # [Optimization] Pre-calculate masks on the target DEVICE once per evaluation call.
    # This prevents repeated device transfers in the loop.
    hist_coo = user_history_sp.coalesce().to(device)
    hist_indices = hist_coo.indices() # (2, nnz)
    
    gt_coo = test_gt_sp.coalesce().to(device)
    gt_indices = gt_coo.indices() # (2, nnz)
    
    # Linear Index for vectorized hit detection: user_id * n_items + item_id
    gt_linear_idx = gt_indices[0] * n_items + gt_indices[1]
    
    sums = defaultdict(float)
    counts = defaultdict(float)

    batch_size = test_loader.batch_size
    with torch.no_grad():
        for start in tqdm(range(0, n_users, batch_size), desc="Eval (Vectorized)", file=sys.stdout):
            end = min(start + batch_size, n_users)
            u_ids = torch.arange(start, end, device=device)
            B = end - start

            # [GPU] 1. Inference
            scores = model.forward(u_ids) # (B, I)

            # [GPU] 2. Vectorized Masking (Already on device, no copy!)
            # Filter history entries belonging to current batch
            h_mask = (hist_indices[0] >= start) & (hist_indices[0] < end)
            if h_mask.any():
                scores[hist_indices[0, h_mask] - start, hist_indices[1, h_mask]] = -1e10

            # [GPU] 3. Top-K and Hit detection
            _, top_idx = torch.topk(scores, k=max_k, dim=1)
            
            # Linear indices for recommendations
            u_idx_expanded = torch.arange(start, end, device=device).unsqueeze(1).expand(-1, max_k)
            top_linear_idx = u_idx_expanded * n_items + top_idx
            
            # [GPU] Vectorized isin - Extremely fast on GPU
            hit_mat = torch.isin(top_linear_idx, gt_linear_idx) # (B, max_k)
            
            # [GPU] 4. GT sizes for Recall/NDCG
            b_gt_mask = (gt_indices[0] >= start) & (gt_indices[0] < end)
            gt_sizes = torch.zeros(B, device=device)
            if b_gt_mask.any():
                gt_sizes.scatter_add_(0, gt_indices[0, b_gt_mask] - start, torch.ones(b_gt_mask.sum(), device=device))

            # [GPU] 5. Metric calculation in batch
            for k in top_k_list:
                hk = hit_mat[:, :k].float()
                hit_sum = hk.sum(dim=1)
                
                if 'Recall' in metrics_list:
                    sums[f'Recall@{k}'] += (hit_sum / gt_sizes.clamp(min=1)).sum().item()
                if 'HitRate' in metrics_list:
                    sums[f'HitRate@{k}'] += (hit_sum > 0).float().sum().item()
                if 'NDCG' in metrics_list:
                    log2_w = torch.tensor(_LOG2_RECIP[:k], device=device)
                    dcg = (hk * log2_w).sum(dim=1)
                    # Ideal DCG calculation
                    idcg = torch.zeros(B, device=device)
                    for i in range(B):
                        idcg[i] = _LOG2_RECIP[:int(min(k, gt_sizes[i].item()))].sum()
                    sums[f'NDCG@{k}'] += (dcg / idcg.clamp(min=1e-8)).sum().item()
            
            counts['total'] += B

    # Final Average
    final = {}
    n = counts['total'] + 1e-12
    for k in top_k_list:
        for m in ['HitRate', 'Recall', 'NDCG']:
            key = f'{m}@{k}'
            final[key] = sums[key] / n
    return final

def evaluate_metrics(model, data_loader, eval_config, device, test_loader, is_final=False):
    history_sp = data_loader.sp_train_valid if is_final else data_loader.sp_train
    test_gt_sp = data_loader.sp_test_gt if is_final else data_loader.sp_valid_gt

    # Called by trainer
    return _evaluate_full(
        model, test_loader, 
        top_k_list=eval_config.get('top_k', [10, 20]), 
        metrics_list=eval_config.get('metrics', ['Recall', 'NDCG']), 
        device=device, 
        user_history_sp=history_sp, 
        test_gt_sp=test_gt_sp
    )
