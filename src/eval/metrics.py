import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

_LOG2_RECIP = (1.0 / np.log2(np.arange(2, 10002, dtype=np.float64))).astype(np.float32)

def get_ndcg(pred_list, ground_truth):
    if not ground_truth: return 0.0
    gt_set = set(ground_truth)
    hits = np.array([1 if item in gt_set else 0 for item in pred_list], dtype=np.float32)
    dcg = float(np.dot(hits, _LOG2_RECIP[:len(pred_list)]))
    idcg = float(_LOG2_RECIP[:min(len(gt_set), len(pred_list))].sum())
    return dcg / idcg if idcg > 0 else 0.0

def get_recall(pred_list, ground_truth):
    if not ground_truth: return 0.0
    return len(set(pred_list).intersection(set(ground_truth))) / len(ground_truth)

def get_hit_rate(pred_list, ground_truth):
    return 1 if set(pred_list).intersection(set(ground_truth)) else 0

def evaluate_metrics(model, data_loader, eval_config, device, test_loader, is_final=False):
    model.eval()
    top_k_list = eval_config.get('top_k', [10])
    metrics_list = eval_config.get('metrics', ['NDCG', 'Recall', 'HitRate'])
    user_history = data_loader.eval_user_history if is_final else data_loader.train_user_history
    
    # Collect ground truth
    user_gt = defaultdict(list)
    for u_batch, i_batch in test_loader:
        for u, i in zip(u_batch.numpy(), i_batch.numpy()): user_gt[u].append(i)
    
    unique_users = sorted(list(user_gt.keys()))
    results = {f'{m}@{k}': [] for k in top_k_list for m in metrics_list}
    
    batch_size = test_loader.batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_users), batch_size), desc="Eval"):
            u_ids = unique_users[i:i+batch_size]
            u_tensor = torch.LongTensor(u_ids).to(device)
            scores = model.forward(u_tensor)
            
            # Masking
            for idx, u_id in enumerate(u_ids):
                seen = user_history.get(u_id, set())
                gt = set(user_gt[u_id])
                exclude = [it for it in seen if it not in gt]
                if exclude: scores[idx, exclude] = -1e9
            
            _, top_indices = torch.topk(scores, k=max(top_k_list), dim=1)
            top_indices = top_indices.cpu().numpy()
            
            for idx, u_id in enumerate(u_ids):
                pred_list = top_indices[idx].tolist()
                gt = user_gt[u_id]
                for k in top_k_list:
                    pk = pred_list[:k]
                    if 'NDCG' in metrics_list: results[f'NDCG@{k}'].append(get_ndcg(pk, gt))
                    if 'Recall' in metrics_list: results[f'Recall@{k}'].append(get_recall(pk, gt))
                    if 'HitRate' in metrics_list: results[f'HitRate@{k}'].append(get_hit_rate(pk, gt))
                    
    return {k: np.mean(v) for k, v in results.items()}
