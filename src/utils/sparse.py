import numpy as np
from scipy import sparse
import torch
import gc

# Global cache to store pre-computed matrices across HPO trials
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

def compute_gram_matrix(X, data_loader=None, weights=None, item_weights=None, block_size=4000):
    """
    Computes G = (D_i) X.T @ diag(W_u) @ X (D_i) as a DENSE float32 matrix.
    Uses Block-wise computation to avoid SciPy's sparse-sparse product memory explosion (83GB peak).
    
    Memory overhead: Final Dense G (8.3GB) + 2x Small Blocks (~1GB) = ~10GB Max.
    """
    dataset_name = getattr(data_loader, 'dataset_name', 'default') if data_loader else 'unknown'
    n_users, n_items = X.shape
    
    print(f"[SparseUtil] Computing Block-wise Dense Gram for {dataset_name} ({n_items} items) on CPU (No Caching)...")
    
    # 2. Prepare Result Matrix (Pre-allocate 8.3GB for float32)
    G = np.zeros((n_items, n_items), dtype=np.float32)
    
    # Efficient Slicing requires CSC
    X_csc = X.tocsc()
    # Transpose for fast Sparse-Dense multiplication (CSR @ Dense is optimized)
    X_T_csr = X_csc.T.tocsr()
    
    # Force weights to float32
    if weights is not None:
        weights = weights.astype(np.float32).reshape(-1, 1)
    if item_weights is not None:
        item_weights = item_weights.astype(np.float32)

    # 3. Block-wise filling (This avoids 83GB peak)
    for start in range(0, n_items, block_size):
        end = min(start + block_size, n_items)
        
        # (Users, Block) dense slice - approx 0.8GB for block_size=5000
        X_block = X_csc[:, start:end].toarray().astype(np.float32)
        
        # Apply User Weights (Propensity/Adaptive)
        if weights is not None:
            X_block *= weights
            
        # G_block = X.T @ X_block
        # CSR @ Dense is handled by optimized BLAS (Standard in NumPy/SciPy)
        G_block = (X_T_csr @ X_block).astype(np.float32)
        
        # Apply Item Weights (Normalization)
        if item_weights is not None:
            # Row-wise scale: G[i, j] *= item_weights[i]
            # Column-wise scale: G[i, j] *= item_weights[j]
            G_block *= item_weights[:, np.newaxis] # Item weights for entire items
            G_block *= item_weights[start:end]     # Item weights for current block
            
        G[:, start:end] = G_block
        
        del X_block, G_block
        if start % (block_size * 4) == 0:
            gc.collect()

    print(f"[SparseUtil] Dense Gram computation complete. Peak RAM used: ~{10 + (n_items*n_users*4/1e9):.1f}GB")
    
    return G
