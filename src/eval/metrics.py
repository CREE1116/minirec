import torch
import numpy as np
from tqdm import tqdm
import sys
from collections import defaultdict

# Precomputed weights for NDCG on CPU (Numpy)
_MAX_K = 1000
_LOG2_RECIP = (1.0 / np.log2(np.arange(2, _MAX_K + 2))).astype(np.float32)
_IDCG_TABLE = np.cumsum(_LOG2_RECIP)

def _evaluate_full(model, test_loader, top_k_list, metrics_list, device, data_loader, is_final):
    # CLAE Style: Use pre-calculated dicts from data_loader
    gt_dict = data_loader.test_gt_dict if is_final else data_loader.valid_gt_dict
    history_dict = data_loader.eval_user_history if is_final else data_loader.train_user_history
    
    n_users = data_loader.n_users
    n_items = data_loader.n_items
    max_k = max(top_k_list)
    
    sums = defaultdict(float)
    total_valid_users = 0
    
    batch_size = test_loader.batch_size
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Eval (CLAE Style)", file=sys.stdout):
            u_ids = batch[0].to(device)
            u_ids_np = u_ids.cpu().numpy()
            B = len(u_ids_np)

            # 1. Inference (GPU)
            scores = model.forward(u_ids) # (B, I)

            # 2. Mask History (GPU)
            # CLAE style: Collect all items to mask in this batch
            exclude_u = []
            exclude_i = []
            for i, u in enumerate(u_ids_np):
                items = history_dict.get(int(u), [])
                exclude_u.extend([i] * len(items))
                exclude_i.extend(items)
            
            if exclude_u:
                scores[exclude_u, exclude_i] = -1e10

            # 3. Top-K extraction (GPU)
            _, top_idx_gpu = torch.topk(scores, k=max_k, dim=1)
            
            # 4. Transfer result to CPU
            top_idx = top_idx_gpu.cpu().numpy()
            
            # 5. Metric Calculation (CPU - Set lookup)
            hits = np.zeros((B, max_k), dtype=np.float32)
            gt_lens = np.zeros(B, dtype=np.int32)
            
            batch_has_gt = []
            for i, u in enumerate(u_ids_np):
                u_gt = gt_dict.get(int(u), set())
                if len(u_gt) > 0:
                    gt_lens[i] = len(u_gt)
                    batch_has_gt.append(i)
                    for rank, item in enumerate(top_idx[i]):
                        if item in u_gt:
                            hits[i, rank] = 1.0
            
            if not batch_has_gt:
                continue
                
            v_idx = np.array(batch_has_gt)
            v_hits = hits[v_idx]
            v_gt_lens = gt_lens[v_idx]
            total_valid_users += len(v_idx)

            # Vectorized metrics via Numpy
            for k in top_k_list:
                hk = v_hits[:, :k]
                hit_count = hk.sum(axis=1)
                
                if 'Recall' in metrics_list:
                    sums[f'Recall@{k}'] += np.sum(hit_count / np.minimum(k, v_gt_lens))
                
                if 'HitRate' in metrics_list:
                    sums[f'HitRate@{k}'] += np.sum(hit_count > 0)
                    
                if 'NDCG' in metrics_list:
                    dcg = np.sum(hk * _LOG2_RECIP[:k], axis=1)
                    # Use pre-calculated IDCG table
                    idcg = _IDCG_TABLE[np.minimum(k, v_gt_lens) - 1]
                    sums[f'NDCG@{k}'] += np.sum(dcg / np.maximum(idcg, 1e-8))

    # 6. Final Average
    final = {}
    n = total_valid_users + 1e-12
    for k in top_k_list:
        for m in ['HitRate', 'Recall', 'NDCG']:
            key = f'{m}@{k}'
            if key in sums:
                final[key] = float(sums[key] / n)
    return final

def evaluate_metrics(model, data_loader, eval_config, device, test_loader, is_final=False):
    # This is the entry point called by Trainer
    return _evaluate_full(
        model, test_loader, 
        top_k_list=eval_config.get('top_k', [10, 20]), 
        metrics_list=eval_config.get('metrics', ['Recall', 'NDCG', 'HitRate']), 
        device=device, 
        data_loader=data_loader,
        is_final=is_final
    )
