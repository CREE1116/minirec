import numpy as np
from scipy import sparse
import torch

# Global cache to store pre-computed matrices across HPO trials
# Key format: (dataset_name, matrix_type, shape)
_GLOBAL_SPARSE_CACHE = {}

def clear_sparse_cache():
    """Clears the global sparse matrix cache to free memory."""
    global _GLOBAL_SPARSE_CACHE
    print("[SparseUtil] Clearing global sparse matrix cache...")
    _GLOBAL_SPARSE_CACHE.clear()

def get_train_matrix_scipy(data_loader):
    """Returns a Scipy CSR matrix for efficient operations, with caching support."""
    dataset_name = getattr(data_loader, 'dataset_name', 'default')
    shape = (data_loader.n_users, data_loader.n_items)
    cache_key = (dataset_name, 'X', shape)
    
    if cache_key in _GLOBAL_SPARSE_CACHE:
        return _GLOBAL_SPARSE_CACHE[cache_key]
    
    print(f"[SparseUtil] Creating Scipy CSR matrix for {dataset_name} {shape}...")
    # DataLoader에서 데이터프레임 대신 numpy 배열을 사용하도록 변경됨
    row = data_loader.train_users
    col = data_loader.train_items
    data = np.ones(len(row), dtype=np.float32)
    X = sparse.csr_matrix((data, (row, col)), shape=shape, dtype=np.float32)
    
    _GLOBAL_SPARSE_CACHE[cache_key] = X
    return X

def compute_gram_matrix(X, data_loader=None):
    """Computes X.T @ X and caches it as a SPARSE matrix to save memory.
    Returns a DENSE copy (toarray) for calculation.
    """
    dataset_name = getattr(data_loader, 'dataset_name', 'default') if data_loader else 'unknown'
    shape = X.shape # (Users, Items)
    cache_key = (dataset_name, 'G_sparse', shape)
    
    if cache_key in _GLOBAL_SPARSE_CACHE:
        # Sparse 형태로 저장된 것을 가져와서 dense로 변환하여 반환
        return _GLOBAL_SPARSE_CACHE[cache_key].toarray().astype(np.float32)
    
    print(f"[SparseUtil] Computing Gram matrix (X.T @ X) for {dataset_name} {shape}...")
    # sparse dot product 수행 (X.T @ X)
    G_sparse = X.T.dot(X)
    
    # Sparse 형태로 캐싱
    _GLOBAL_SPARSE_CACHE[cache_key] = G_sparse
    
    # Dense 형태로 반환하여 즉시 사용 가능하게 함
    return G_sparse.toarray().astype(np.float32)
