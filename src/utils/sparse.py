import numpy as np
from scipy import sparse
import torch

# Global cache to store pre-computed matrices across HPO trials
# Key format: (dataset_name, matrix_type, shape)
_GLOBAL_SPARSE_CACHE = {}

def get_train_matrix_scipy(data_loader):
    """Returns a Scipy CSR matrix for efficient operations, with caching support."""
    dataset_name = getattr(data_loader, 'dataset_name', 'default')
    shape = (data_loader.n_users, data_loader.n_items)
    cache_key = (dataset_name, 'X', shape)
    
    if cache_key in _GLOBAL_SPARSE_CACHE:
        return _GLOBAL_SPARSE_CACHE[cache_key]
    
    print(f"[SparseUtil] Creating Scipy CSR matrix for {dataset_name} {shape}...")
    train_df = data_loader.train_df
    row = train_df['user_id'].values
    col = train_df['item_id'].values
    data = np.ones(len(train_df), dtype=np.float32)
    X = sparse.csr_matrix((data, (row, col)), shape=shape, dtype=np.float32)
    
    _GLOBAL_SPARSE_CACHE[cache_key] = X
    return X

def compute_gram_matrix(X, data_loader=None):
    """Computes X.T @ X using Scipy efficiently, with caching support.
    Returns a COPY of the matrix to prevent in-place modification bugs.
    """
    dataset_name = getattr(data_loader, 'dataset_name', 'default') if data_loader else 'unknown'
    shape = X.shape # (Users, Items)
    # X.T @ X 의 결과는 (Items, Items)
    cache_key = (dataset_name, 'G', shape)
    
    if cache_key in _GLOBAL_SPARSE_CACHE:
        return _GLOBAL_SPARSE_CACHE[cache_key].copy()
    
    print(f"[SparseUtil] Computing Gram matrix (X.T @ X) for {dataset_name} {shape}...")
    # .astype(np.float32)를 추가하여 확실하게 float32 유지
    G = X.T.dot(X).toarray().astype(np.float32)
    
    _GLOBAL_SPARSE_CACHE[cache_key] = G
    
    return G.copy()
