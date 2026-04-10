import numpy as np
from scipy import sparse
import torch

# Global cache to store pre-computed matrices across HPO trials
_GLOBAL_SPARSE_CACHE = {}

def get_train_matrix_scipy(data_loader):
    """Returns a Scipy CSR matrix for efficient operations, with caching support."""
    # Use DataLoader's cache_filename as key so different instances hit the same cache
    cache_key = getattr(data_loader, 'cache_filename', id(data_loader))
    
    if cache_key in _GLOBAL_SPARSE_CACHE and 'X' in _GLOBAL_SPARSE_CACHE[cache_key]:
        return _GLOBAL_SPARSE_CACHE[cache_key]['X']
    
    train_df = data_loader.train_df
    row = train_df['user_id'].values
    col = train_df['item_id'].values
    data = np.ones(len(train_df), dtype=np.float32)
    X = sparse.csr_matrix((data, (row, col)), shape=(data_loader.n_users, data_loader.n_items))
    
    if cache_key not in _GLOBAL_SPARSE_CACHE:
        _GLOBAL_SPARSE_CACHE[cache_key] = {}
    _GLOBAL_SPARSE_CACHE[cache_key]['X'] = X
    return X

def compute_gram_matrix(X, data_loader=None):
    """Computes X.T @ X using Scipy efficiently, with caching support."""
    # Use cache_filename from loader if available, otherwise fallback to object ID
    cache_key = getattr(data_loader, 'cache_filename', id(X))
    
    if cache_key in _GLOBAL_SPARSE_CACHE and 'G' in _GLOBAL_SPARSE_CACHE[cache_key]:
        return _GLOBAL_SPARSE_CACHE[cache_key]['G']
    
    G = X.T.dot(X).toarray()
    
    if cache_key not in _GLOBAL_SPARSE_CACHE:
        _GLOBAL_SPARSE_CACHE[cache_key] = {}
    _GLOBAL_SPARSE_CACHE[cache_key]['G'] = G
    return G
