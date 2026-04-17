import numpy as np
from scipy import sparse
import torch
import gc

# Global cache to store ONLY sparse matrices (they are small)
_GLOBAL_SPARSE_CACHE = {}

def clear_sparse_cache():
    """Clears the global sparse matrix cache to free memory."""
    global _GLOBAL_SPARSE_CACHE
    print("[SparseUtil] Clearing global sparse matrix cache...")
    _GLOBAL_SPARSE_CACHE.clear()
    gc.collect()

def get_train_matrix_scipy(data_loader):
    """Returns a Scipy CSR matrix for efficient operations, with caching support."""
    dataset_name = getattr(data_loader, 'dataset_name', 'default')
    shape = (data_loader.n_users, data_loader.n_items)
    cache_key = (dataset_name, 'X', shape)
    
    if cache_key in _GLOBAL_SPARSE_CACHE:
        return _GLOBAL_SPARSE_CACHE[cache_key]
    
    print(f"[SparseUtil] Creating Scipy CSR matrix for {dataset_name} {shape}...")
    row = data_loader.train_users.astype(np.int32)
    col = data_loader.train_items.astype(np.int32)
    data = np.ones(len(row), dtype=np.float32)
    X = sparse.csr_matrix((data, (row, col)), shape=shape, dtype=np.float32)
    
    _GLOBAL_SPARSE_CACHE[cache_key] = X
    return X

def compute_gram_matrix(X, data_loader=None, weights=None, item_weights=None):
    """
    Simplest possible Gram Matrix construction (Direct like CLAE).
    Avoids hidden sparse-sparse copies.
    """
    if weights is not None:
        # X_weighted = diag(sqrt(W)) @ X
        # We perform multiplication without creating another full sparse object if possible
        X = X.multiply(np.sqrt(weights).reshape(-1, 1).astype(np.float32))
    
    print(f"  [SparseUtil] Sparse dot product (X.T @ X)...")
    G_sp = X.T.dot(X)
    
    print(f"  [SparseUtil] Converting to Dense float32...")
    # [Critical] Single direct allocation
    G = G_sp.toarray().astype(np.float32)
    del G_sp
    
    if item_weights is not None:
        item_weights = item_weights.astype(np.float32)
        G *= item_weights[:, np.newaxis]
        G *= item_weights[np.newaxis, :]
        
    return G
