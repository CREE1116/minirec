import numpy as np
from scipy import sparse
import torch
import gc

# Global cache to store ONLY sparse matrices
_GLOBAL_SPARSE_CACHE = {}

def clear_sparse_cache():
    """Clears the global sparse matrix cache to free memory."""
    global _GLOBAL_SPARSE_CACHE
    print("[SparseUtil] Clearing global sparse matrix cache...")
    _GLOBAL_SPARSE_CACHE.clear()
    gc.collect()

def get_train_matrix_scipy(data_loader):
    """Returns a Scipy CSR matrix in float32."""
    dataset_name = getattr(data_loader, 'dataset_name', 'default')
    shape = (data_loader.n_users, data_loader.n_items)
    cache_key = (dataset_name, 'X', shape)
    
    if cache_key in _GLOBAL_SPARSE_CACHE:
        return _GLOBAL_SPARSE_CACHE[cache_key]
    
    print(f"[SparseUtil] Creating Scipy CSR matrix (float32) for {dataset_name}...")
    row = data_loader.train_users.astype(np.int32)
    col = data_loader.train_items.astype(np.int32)
    # 데이터 생성 시점부터 float32로 고정
    data = np.ones(len(row), dtype=np.float32)
    X = sparse.csr_matrix((data, (row, col)), shape=shape, dtype=np.float32)
    
    _GLOBAL_SPARSE_CACHE[cache_key] = X
    return X

def compute_gram_matrix(X, data_loader=None, weights=None, item_weights=None):
    # 0. 시작하자마자 X의 타입을 체크/변경 (X가 Sparse라면 astype은 효율적입니다)
    if X.dtype != np.float32:
        X = X.astype(np.float32)

    # 1. User weighting
    if weights is not None:
        # np.sqrt의 결과를 명시적으로 float32로
        w = np.sqrt(weights, dtype=np.float32).reshape(-1, 1)
        X = X.multiply(w)
    
    # 2. Gram matrix 계산 및 즉시 타입 확인
    G_sp = X.T.dot(X)
    if G_sp.dtype != np.float32:
        G_sp = G_sp.astype(np.float32)
    
    # 3. Dense 변환 (7.5GB 할당)
    G = G_sp.toarray()
    del G_sp
    
    # 4. Item weighting (item_weights의 타입을 float32로 강제)
    if item_weights is not None:
        # item_weights가 float64면 G가 float64로 변하며 복사본이 생깁니다.
        iw = item_weights.astype(np.float32)
        G *= iw[:, np.newaxis]
        G *= iw[np.newaxis, :]
        
    return G
