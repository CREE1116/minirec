import os
import pickle
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader as PyTorchDataLoader, TensorDataset

# ============================================================
# 데이터 전처리 핵심 유틸리티 (분할 방식 정돈)
# ============================================================

def load_raw_data(data_path, separator, columns, has_header=False, format='csv'):
    if format == 'lightgcn':
        users, items = [], []
        with open(data_path, 'r') as f:
            for line in f:
                parts = line.strip().split(separator)
                if not parts: continue
                u = parts[0]
                for i in parts[1:]:
                    users.append(u)
                    items.append(i)
        return pd.DataFrame({columns[0]: users, columns[1]: items})
    
    engine = 'c'
    if len(separator) > 1 and separator != r'\s+': engine = 'python'
    kwargs = {'sep': separator, 'engine': engine}
    if has_header:
        df = pd.read_csv(data_path, header=0, **kwargs)
        df.columns = columns[:len(df.columns)]
    else:
        df = pd.read_csv(data_path, header=None, names=columns, **kwargs)
    return df

def filter_interactions(df, min_u, min_i):
    curr_len = len(df)
    while True:
        u_counts = df['user_id'].value_counts()
        df = df[df['user_id'].isin(u_counts[u_counts >= min_u].index)]
        i_counts = df['item_id'].value_counts()
        df = df[df['item_id'].isin(i_counts[i_counts >= min_i].index)]
        if len(df) == curr_len: break
        curr_len = len(df)
    return df

def remap_ids(df):
    u_list, i_list = sorted(df['user_id'].unique()), sorted(df['item_id'].unique())
    user_map = {old: new for new, old in enumerate(u_list)}
    item_map = {old: new for new, old in enumerate(i_list)}
    df = df.copy()
    df['user_id'] = pd.Categorical(df['user_id'], categories=u_list).codes.astype(np.int64)
    df['item_id'] = pd.Categorical(df['item_id'], categories=i_list).codes.astype(np.int64)
    return df, user_map, item_map, len(u_list), len(i_list)

# 1. LOO (Leave-One-Out)
def split_loo(df):
    df_sorted = df.sort_values(by=['user_id', 'timestamp', 'item_id']) if 'timestamp' in df.columns else df.sort_values(by=['user_id', 'item_id'])
    df_sorted = df_sorted.copy()
    df_sorted['_rank'] = df_sorted.groupby('user_id', sort=False).cumcount(ascending=False)
    return df_sorted[df_sorted['_rank'] >= 2].drop(columns=['_rank']), \
           df_sorted[df_sorted['_rank'] == 1].drop(columns=['_rank']), \
           df_sorted[df_sorted['_rank'] == 0].drop(columns=['_rank'])

# 2. RS (Random Split - Ratio based)
def split_random(df, train_ratio=0.8, valid_ratio=0.1, seed=42):
    if len(df) == 0: return df.copy(), df.copy(), df.copy()
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_shuffled = df_shuffled.sort_values(by='user_id', kind='stable').copy()
    
    total_counts = df_shuffled.groupby('user_id', sort=False).size()
    cum_counts = df_shuffled.groupby('user_id', sort=False).cumcount().values
    total_counts_v = total_counts.loc[df_shuffled['user_id']].values
    
    train_end = np.clip((total_counts_v * train_ratio).astype(int), 1, None)
    valid_end = (total_counts_v * (train_ratio + valid_ratio)).astype(int)
    
    mask_lt3 = total_counts_v < 3
    valid_end = np.where((total_counts_v >= 3) & (valid_end >= total_counts_v), total_counts_v - 1, valid_end)
    train_end = np.where((total_counts_v >= 3) & (train_end >= valid_end), valid_end - 1, train_end)
    
    return df_shuffled[(cum_counts < train_end) | mask_lt3].copy(), \
           df_shuffled[(~mask_lt3) & (cum_counts >= train_end) & (cum_counts < valid_end)].copy(), \
           df_shuffled[(~mask_lt3) & (cum_counts >= valid_end)].copy()

# 3. Temporal RS (Temporal Ratio based)
def split_temporal_ratio(df, train_ratio=0.8, valid_ratio=0.1):
    if len(df) == 0: return df.copy(), df.copy(), df.copy()
    # 시간순 정렬 필수
    df_sorted = df.sort_values(by=['user_id', 'timestamp', 'item_id']).copy() if 'timestamp' in df.columns else df.sort_values(by=['user_id', 'item_id']).copy()
    
    total_counts = df_sorted.groupby('user_id', sort=False).size()
    cum_counts = df_sorted.groupby('user_id', sort=False).cumcount().values
    total_counts_v = total_counts.loc[df_sorted['user_id']].values
    
    train_end = np.clip((total_counts_v * train_ratio).astype(int), 1, None)
    valid_end = (total_counts_v * (train_ratio + valid_ratio)).astype(int)
    
    mask_lt3 = total_counts_v < 3
    valid_end = np.where((total_counts_v >= 3) & (valid_end >= total_counts_v), total_counts_v - 1, valid_end)
    train_end = np.where((total_counts_v >= 3) & (train_end >= valid_end), valid_end - 1, train_end)
    
    return df_sorted[(cum_counts < train_end) | mask_lt3].copy(), \
           df_sorted[(~mask_lt3) & (cum_counts >= train_end) & (cum_counts < valid_end)].copy(), \
           df_sorted[(~mask_lt3) & (cum_counts >= valid_end)].copy()

# ============================================================
# DataLoader 메인 클래스
# ============================================================

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.data_cache_path = config.get('data_cache_path', './data_cache/')
        os.makedirs(self.data_cache_path, exist_ok=True)
        
        self.cache_filename = self._get_cache_filename()
        self.cache_path = os.path.join(self.data_cache_path, self.cache_filename)
        
        if os.path.exists(self.cache_path):
            self._load_from_cache()
        else:
            self._process_data()
            self._save_to_cache()

    def _get_cache_filename(self):
        d_name = self.config.get('dataset_name', 'data')
        min_u = self.config.get('min_user_interactions', 5)
        min_i = self.config.get('min_item_interactions', 5)
        split = self.config.get('split_method', 'loo')
        rt = self.config.get('rating_threshold', 0)
        tr = self.config.get('train_ratio', 0.8)
        vr = self.config.get('valid_ratio', 0.1)
        seed = self.config.get('seed', 42)
        fmt = self.config.get('format', 'csv')
        return f"{d_name}_u{min_u}_i{min_i}_{split}_rt{rt}_tr{tr}_vr{vr}_s{seed}_{fmt}.pkl"

    def _process_data(self):
        print(f"[DataLoader] Processing data for: {self.config.get('dataset_name')}...")
        
        # Check if pre-split paths are provided
        train_path = self.config.get('train_path')
        valid_path = self.config.get('valid_path')
        test_path = self.config.get('test_path')
        fmt = self.config.get('format', 'csv')

        if train_path:
            print(f"  loading pre-split data (format: {fmt})...")
            sep = self.config.get('separator', ',')
            cols = self.config['columns']
            has_header = self.config.get('has_header', False)
            
            train_df = load_raw_data(train_path, sep, cols, has_header, format=fmt)
            valid_df = load_raw_data(valid_path, sep, cols, has_header, format=fmt) if valid_path else pd.DataFrame(columns=cols)
            test_df = load_raw_data(test_path, sep, cols, has_header, format=fmt) if test_path else pd.DataFrame(columns=cols)
            
            print(f"  raw train={len(train_df):,}, valid={len(valid_df):,}, test={len(test_df):,}")
            
            # Combine all to get consistent ID mapping
            full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
            _, self.user_map, self.item_map, self.n_users, self.n_items = remap_ids(full_df)
            
            def map_df(df):
                if len(df) == 0: return df.copy()
                df = df.copy()
                df['user_id'] = df['user_id'].map(self.user_map).astype(np.int64)
                df['item_id'] = df['item_id'].map(self.item_map).astype(np.int64)
                return df
            
            self.train_df = map_df(train_df)
            self.valid_df = map_df(valid_df)
            self.test_df = map_df(test_df)
            self.df = pd.concat([self.train_df, self.valid_df, self.test_df], ignore_index=True)
            
            density = len(self.df) / (self.n_users * self.n_items) * 100
            print(f"  users={self.n_users:,}  items={self.n_items:,}  interactions={len(self.df):,}  density={density:.4f}%")
            
            split_method = 'pre-split'
        else:
            # Original loading and splitting logic
            df = load_raw_data(self.config['data_path'], self.config['separator'], self.config['columns'], self.config.get('has_header', False), format=fmt)
            n_raw = len(df)
            print(f"  raw interactions : {n_raw:,}")

            # 컬럼 이름 식별
            col_names = self.config['columns']
            rating_col = next((c for c in col_names if 'rating' in c.lower()), None)
            time_col = next((c for c in col_names if 'timestamp' in c.lower() or 'time' in c.lower()), None)

            # Rating 필터링 (컬럼이 존재할 때만)
            if rating_col and self.config.get('rating_threshold', 0) > 0:
                df = df[df[rating_col] >= self.config['rating_threshold']]
                print(f"  after rating threshold ({self.config['rating_threshold']}) on '{rating_col}': {len(df):,}  (-{n_raw - len(df):,})")
            elif not rating_col and self.config.get('rating_threshold', 0) > 0:
                print(f"  [Warning] rating_threshold set but no 'rating' column found. Skipping rating filter.")

            if self.config.get('dedup', True):
                before = len(df)
                df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
                if before - len(df):
                    print(f"  after dedup      : {len(df):,}  (-{before - len(df):,})")

            before = len(df)
            df = filter_interactions(df, self.config.get('min_user_interactions', 5), self.config.get('min_item_interactions', 5))
            print(f"  after k-core     : {len(df):,}  (-{before - len(df):,})")

            self.df, self.user_map, self.item_map, self.n_users, self.n_items = remap_ids(df)
            if self.n_users == 0 or self.n_items == 0:
                raise ValueError(f"No data left after filtering for dataset: {self.config.get('dataset_name')}. Check your rating_threshold or k-core settings.")
                
            density = len(df) / (self.n_users * self.n_items) * 100
            print(f"  users={self.n_users:,}  items={self.n_items:,}  interactions={len(df):,}  density={density:.4f}%")

            split_method = self.config.get('split_method', 'loo').lower()
            tr = self.config.get('train_ratio', 0.8)
            vr = self.config.get('valid_ratio', 0.1)
            seed = self.config.get('seed', 42)

            # Split Method 검증 및 경고
            is_temporal = 'temporal' in split_method or split_method == 'loo'
            if is_temporal and not time_col:
                print(f"\n[!!! WARNING !!!] Split method '{split_method}' requires 'timestamp' column, but it was NOT found in columns: {col_names}.")
                print(f"Falling back to 'random' split to prevent crash, but results will NOT be chronologically ordered.\n")
                split_method = 'random'

            if split_method == 'loo':
                self.train_df, self.valid_df, self.test_df = split_loo(self.df)
            elif split_method in ['rs', 'random']:
                self.train_df, self.valid_df, self.test_df = split_random(self.df, tr, vr, seed)
            elif split_method in ['temporal', 'temporal_ratio', 'temporal_rs']:
                self.train_df, self.valid_df, self.test_df = split_temporal_ratio(self.df, tr, vr)
            else:
                raise ValueError(f"Unknown split method: {split_method}")

        print(f"  split ({split_method})  train={len(self.train_df):,}  valid={len(self.valid_df):,}  test={len(self.test_df):,}")

        self.train_user_history = self.train_df.groupby('user_id')['item_id'].agg(set).to_dict()
        self.eval_user_history = pd.concat([self.train_df, self.valid_df]).groupby('user_id')['item_id'].agg(set).to_dict()

        train_counts = self.train_df['item_id'].value_counts()
        self.item_popularity = train_counts.reindex(range(self.n_items), fill_value=0).sort_index().values

        self.sampling_weights = None
        if self.config.get('train', {}).get('negative_sampling_strategy') == 'popularity':
            counts = torch.FloatTensor(self.item_popularity)
            self.sampling_weights = torch.pow(counts, self.config['train'].get('negative_sampling_alpha', 0.75))
            self.sampling_weights /= self.sampling_weights.sum()

    def _save_to_cache(self):
        data = {
            'train_df': self.train_df, 'valid_df': self.valid_df, 'test_df': self.test_df,
            'user_map': self.user_map, 'item_map': self.item_map,
            'n_users': self.n_users, 'n_items': self.n_items,
            'item_popularity': self.item_popularity,
            'train_user_history': self.train_user_history,
            'eval_user_history': self.eval_user_history,
            'sampling_weights': self.sampling_weights
        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(data, f)

    def _load_from_cache(self):
        print(f"[DataLoader] Cache hit: {self.cache_path}")
        with open(self.cache_path, 'rb') as f:
            data = pickle.load(f)
        for k, v in data.items():
            setattr(self, k, v)

    def get_interaction_graph(self, add_self_loops=False):
        """LightGCN 등 GCN 모델을 위한 user-item bipartite graph (sparse COO tensor).

        크기: (n_users + n_items) x (n_users + n_items)
        구조:
            [  0   R  ]
            [ R^T  0  ]
        """
        rows = self.train_df['user_id'].values
        cols = self.train_df['item_id'].values + self.n_users  # item 인덱스를 user 뒤로 오프셋
        n = self.n_users + self.n_items

        # 양방향 엣지
        edge_rows = np.concatenate([rows, cols])
        edge_cols = np.concatenate([cols, rows])
        values = np.ones(len(edge_rows), dtype=np.float32)

        indices = torch.LongTensor(np.stack([edge_rows, edge_cols]))
        vals = torch.FloatTensor(values)
        adj = torch.sparse_coo_tensor(indices, vals, (n, n))
        return adj

    def get_train_loader(self, batch_size):
        ds = RecSysDataset(self.train_df, self.n_items, self.train_user_history, self.config.get('train', {}).get('loss_type', 'pairwise'), self.config.get('train', {}).get('num_negatives', 1), self.sampling_weights)
        return PyTorchDataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate_fn)

    def get_validation_loader(self, batch_size):
        ds = TensorDataset(torch.LongTensor(self.valid_df['user_id'].values.copy()), torch.LongTensor(self.valid_df['item_id'].values.copy()))
        return PyTorchDataLoader(ds, batch_size=batch_size, shuffle=False)

    def get_final_loader(self, batch_size):
        ds = TensorDataset(torch.LongTensor(self.test_df['user_id'].values.copy()), torch.LongTensor(self.test_df['item_id'].values.copy()))
        return PyTorchDataLoader(ds, batch_size=batch_size, shuffle=False)

# ============================================================
# Dataset 클래스
# ============================================================

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
        return {'user_id': torch.LongTensor(u_np).unsqueeze(1), 'pos_item_id': torch.LongTensor(i_np).unsqueeze(1), 'neg_item_id': torch.LongTensor(final_neg)}
