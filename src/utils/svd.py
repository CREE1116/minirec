import os
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

def get_svd_cache(data_loader, k_max=None, matrix=None, cache_id="raw"):
    """
    SVD 결과를 캐싱하거나 로드합니다.
    k_max: None일 경우 데이터셋의 최대 가능 차수(Full Rank)로 계산합니다.
    matrix: SVD를 수행할 행렬 (None이면 data_loader에서 raw matrix 생성)
    cache_id: 캐시 식별자 (예: "raw", "normalized")
    """
    cache_dir = getattr(data_loader, 'cache_dir', './data_cache/')
    dataset_name = getattr(data_loader, 'dataset_name', 'default')
    
    if matrix is None:
        rows = data_loader.train_users
        cols = data_loader.train_items
        matrix = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), 
                               shape=(data_loader.n_users, data_loader.n_items), dtype=np.float32)
    
    # svds는 min(shape) - 1 까지만 계산 가능
    full_rank = min(matrix.shape) - 1
    target_k = full_rank if k_max is None else min(k_max, full_rank)
    
    k_label = f"full{target_k}" if k_max is None else f"k{target_k}"
    svd_cache_path = os.path.join(cache_dir, f"svd_{dataset_name}_{cache_id}_{k_label}.pkl")

    if os.path.exists(svd_cache_path):
        print(f"[SVD Cache] Loading cached {cache_id} SVD ({k_label}) for {dataset_name}...")
        try:
            with open(svd_cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[SVD Cache] Failed to load cache: {e}. Recomputing...")

    print(f"[SVD Cache] Computing {cache_id} SVD (target_k={target_k}) for {dataset_name}...")
    
    u, s, vt = svds(matrix, k=target_k)
    
    idx = np.argsort(s)[::-1]
    u, s, vt = u[:, idx], s[idx], vt[idx, :]
    
    svd_data = {'u': u, 's': s, 'vt': vt}
    
    os.makedirs(cache_dir, exist_ok=True)
    with open(svd_cache_path, 'wb') as f:
        pickle.dump(svd_data, f)
    
    print(f"[SVD Cache] Saved {cache_id} SVD to {svd_cache_path}")
    return svd_data
