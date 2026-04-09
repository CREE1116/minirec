import numpy as np
from scipy import sparse
import torch

def get_train_matrix_scipy(data_loader):
    """효율적인 연산을 위한 Scipy CSR 행렬 반환"""
    train_df = data_loader.train_df
    row = train_df['user_id'].values
    col = train_df['item_id'].values
    data = np.ones(len(train_df), dtype=np.float32)
    X = sparse.csr_matrix((data, (row, col)), shape=(data_loader.n_users, data_loader.n_items))
    return X

def compute_gram_matrix(X):
    """X.T @ X 연산을 Scipy를 사용하여 효율적으로 수행"""
    return X.T.dot(X).toarray()
