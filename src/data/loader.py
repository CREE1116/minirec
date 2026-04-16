import os
import pickle
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader as PyTorchDataLoader, TensorDataset

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.dataset_name = config['dataset_name']
        # 하드코딩된 절대 경로를 프로젝트 루트 기준 상대 경로로 변경
        base_preprocessed_path = os.path.join(os.getcwd(), 'data', 'preprocessed')
        self.data_dir = os.path.join(base_preprocessed_path, self.dataset_name)
        
        # 캐시 경로 설정
        self.cache_dir = config.get('data_cache_path', './data_cache/')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, f"{self.dataset_name}_processed.pkl")

        if os.path.exists(self.cache_path):
            self._load_from_cache()
        else:
            self._load_and_process()
            self._save_to_cache()

    def _load_and_process(self):
        print(f"[DataLoader] Loading preprocessed data for {self.dataset_name} from {self.data_dir}")
        
        self.train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        self.valid_df = pd.read_csv(os.path.join(self.data_dir, 'valid.csv'))
        self.test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))

        # Preprocessed data is already mapped to IDs 0...N-1
        self.n_users = int(max(self.train_df['user_id'].max(), self.valid_df['user_id'].max(), self.test_df['user_id'].max()) + 1)
        self.n_items = int(max(self.train_df['item_id'].max(), self.valid_df['item_id'].max(), self.test_df['item_id'].max()) + 1)
        
        self.df = pd.concat([self.train_df, self.valid_df, self.test_df])

        self.train_user_history = self.train_df.groupby('user_id')['item_id'].agg(set).to_dict()
        self.eval_user_history = pd.concat([self.train_df, self.valid_df]).groupby('user_id')['item_id'].agg(set).to_dict()

        train_counts = self.train_df['item_id'].value_counts()
        self.item_popularity = train_counts.reindex(range(self.n_items), fill_value=0).sort_index().values

        self.sampling_weights = None
        if self.config.get('train', {}).get('negative_sampling_strategy') == 'popularity':
            counts = torch.FloatTensor(self.item_popularity)
            self.sampling_weights = torch.pow(counts, self.config['train'].get('negative_sampling_alpha', 0.75))
            self.sampling_weights /= self.sampling_weights.sum()
        
        self._prepare_sparse_tensors()

    def _prepare_sparse_tensors(self):
        """메모리 내 희소 텐서 선행 생성 및 경고 해결"""
        def to_sp(df):
            # torch.tensor()를 직접 사용하여 Non-writable NumPy array 경고 해결
            r = torch.tensor(df['user_id'].values, dtype=torch.long)
            c = torch.tensor(df['item_id'].values, dtype=torch.long)
            v = torch.ones(r.size(0), dtype=torch.float32)
            # check_invariants=False를 통해 Sparse invariant 경고 제거
            return torch.sparse_coo_tensor(torch.stack([r, c]), v, (self.n_users, self.n_items), check_invariants=False).coalesce()

        self.sp_train = to_sp(self.train_df)
        self.sp_train_valid = to_sp(pd.concat([self.train_df, self.valid_df]))
        self.sp_valid_gt = to_sp(self.valid_df)
        self.sp_test_gt = to_sp(self.test_df)

    def get_interaction_graph(self):
        """LightGCN 등 그래프 모델을 위한 인접 행렬 생성"""
        import scipy.sparse as sp
        
        # Create bipartite adjacency matrix
        R = sp.csr_matrix((np.ones(len(self.train_df)), (self.train_df['user_id'], self.train_df['item_id'])),
                          shape=(self.n_users, self.n_items))
        
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        # Normalize: D^-0.5 A D^-0.5
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        norm_adj = norm_adj.tocoo()
        
        # Convert to torch sparse with check_invariants=False
        indices = torch.tensor(np.vstack((norm_adj.row, norm_adj.col)), dtype=torch.long)
        values = torch.tensor(norm_adj.data, dtype=torch.float32)
        shape = torch.Size(norm_adj.shape)
        return torch.sparse_coo_tensor(indices, values, shape, check_invariants=False).coalesce()

    def _save_to_cache(self):
        data = {
            'train_df': self.train_df, 'valid_df': self.valid_df, 'test_df': self.test_df,
            'n_users': self.n_users, 'n_items': self.n_items,
            'item_popularity': self.item_popularity,
            'train_user_history': self.train_user_history,
            'eval_user_history': self.eval_user_history,
            'sampling_weights': self.sampling_weights
        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(data, f)

    def _load_from_cache(self):
        with open(self.cache_path, 'rb') as f:
            data = pickle.load(f)
        for k, v in data.items():
            setattr(self, k, v)
        self.df = pd.concat([self.train_df, self.valid_df, self.test_df])
        self._prepare_sparse_tensors()

    def get_train_loader(self, batch_size):
        ds = RecSysDataset(self.train_df, self.n_items, self.train_user_history, self.config.get('train', {}).get('loss_type', 'pairwise'), self.config.get('train', {}).get('num_negatives', 1), self.sampling_weights)
        return PyTorchDataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate_fn)

    def get_validation_loader(self, batch_size):
        # torch.tensor()로 안전하게 복사하여 경고 해결
        ds = TensorDataset(torch.tensor(self.valid_df['user_id'].values, dtype=torch.long), 
                           torch.tensor(self.valid_df['item_id'].values, dtype=torch.long))
        return PyTorchDataLoader(ds, batch_size=batch_size, shuffle=False)

    def get_final_loader(self, batch_size):
        ds = TensorDataset(torch.tensor(self.test_df['user_id'].values, dtype=torch.long), 
                           torch.tensor(self.test_df['item_id'].values, dtype=torch.long))
        return PyTorchDataLoader(ds, batch_size=batch_size, shuffle=False)

class RecSysDataset(Dataset):
    def __init__(self, df, n_items, user_history, loss_type, num_negatives, sampling_weights=None):
        self.df, self.n_items, self.user_history = df, n_items, user_history
        self.loss_type, self.num_negatives = loss_type, num_negatives
        self.sampling_weights = sampling_weights
        self.users, self.items = df['user_id'].values, df['item_id'].values

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        return (self.users[idx], self.items[idx])

    def collate_fn(self, batch):
        u, i = zip(*batch)
        u_np, i_np = np.array(u), np.array(i)
        B, N = len(u_np), self.num_negatives
        
        if self.sampling_weights is not None:
            all_neg = np.random.choice(self.n_items, size=(B, N * 3), p=self.sampling_weights.numpy())
        else:
            all_neg = np.random.randint(0, self.n_items, size=(B, N * 3))

        final_neg = np.zeros((B, N), dtype=np.int64)
        for idx in range(B):
            seen = self.user_history.get(int(u_np[idx]), set())
            valid = [c for c in all_neg[idx] if c != int(i_np[idx]) and c not in seen]
            if len(valid) >= N: final_neg[idx] = valid[:N]
            else:
                final_neg[idx, :len(valid)] = valid
                for j in range(len(valid), N):
                    while True:
                        s = np.random.randint(0, self.n_items)
                        if s != int(i_np[idx]) and s not in seen:
                            final_neg[idx, j] = s
                            break
        return {'user_id': torch.tensor(u_np, dtype=torch.long).unsqueeze(1), 'pos_item_id': torch.tensor(i_np, dtype=torch.long).unsqueeze(1), 'neg_item_id': torch.tensor(final_neg, dtype=torch.long)}
