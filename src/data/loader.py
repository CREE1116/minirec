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
        base_preprocessed_path = os.path.join(os.getcwd(), 'data', 'preprocessed')
        self.data_dir = os.path.join(base_preprocessed_path, self.dataset_name)
        
        self.cache_dir = config.get('data_cache_path', './data_cache/')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, f"{self.dataset_name}_processed.pkl")

        if os.path.exists(self.cache_path):
            self._load_from_cache()
        else:
            self._load_and_process()
            self._save_to_cache()

    def _load_and_process(self):
        print(f"[DataLoader] Loading and caching data for {self.dataset_name}...")
        train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        valid_df = pd.read_csv(os.path.join(self.data_dir, 'valid.csv'))
        test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))

        self.n_users = int(max(train_df['user_id'].max(), valid_df['user_id'].max(), test_df['user_id'].max()) + 1)
        self.n_items = int(max(train_df['item_id'].max(), valid_df['item_id'].max(), test_df['item_id'].max()) + 1)
        
        # 1. 히스토리 캐싱 (CLAE style: list of numpy arrays)
        self.train_user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
        self.eval_user_history = pd.concat([train_df, valid_df]).groupby('user_id')['item_id'].apply(list).to_dict()

        # 2. 정답지 캐싱
        self.valid_gt_dict = valid_df.groupby('user_id')['item_id'].apply(set).to_dict()
        self.test_gt_dict = test_df.groupby('user_id')['item_id'].apply(set).to_dict()

        train_counts = train_df['item_id'].value_counts()
        self.item_popularity = train_counts.reindex(range(self.n_items), fill_value=0).sort_index().values

        # 3. Memory Optimization: Store only necessary numpy arrays and delete DFs
        self.train_users = train_df['user_id'].values.astype(np.int32)
        self.train_items = train_df['item_id'].values.astype(np.int32)

        self.sampling_weights = None
        if self.config.get('train', {}).get('negative_sampling_strategy') == 'popularity':
            counts = torch.FloatTensor(self.item_popularity)
            self.sampling_weights = torch.pow(counts, self.config['train'].get('negative_sampling_alpha', 0.75))
            self.sampling_weights /= self.sampling_weights.sum()
        
        # Original DataFrames are no longer needed
        del train_df, valid_df, test_df

    def _save_to_cache(self):
        data = {
            'n_users': self.n_users, 'n_items': self.n_items,
            'item_popularity': self.item_popularity,
            'train_user_history': self.train_user_history,
            'eval_user_history': self.eval_user_history,
            'valid_gt_dict': self.valid_gt_dict,
            'test_gt_dict': self.test_gt_dict,
            'sampling_weights': self.sampling_weights,
            'train_users': self.train_users,
            'train_items': self.train_items
        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(data, f)

    def _load_from_cache(self):
        with open(self.cache_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if all required attributes for the new CLAE-style evaluation exist
        required_keys = ['valid_gt_dict', 'test_gt_dict', 'train_user_history', 'eval_user_history', 'train_users', 'train_items']
        all_present = all(k in data for k in required_keys)
        
        if not all_present:
            print(f"[DataLoader] Cache is outdated. Re-processing...")
            self._load_and_process()
            self._save_to_cache()
            return

        for k, v in data.items():
            setattr(self, k, v)

    def get_train_loader(self, batch_size):
        ds = RecSysDataset(self.train_users, self.train_items, self.n_items, self.train_user_history, self.config.get('train', {}).get('loss_type', 'pairwise'), self.config.get('train', {}).get('num_negatives', 1), self.sampling_weights)
        return PyTorchDataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate_fn)

    def get_validation_loader(self, batch_size):
        # Only users who have validation data
        val_users = np.array(list(self.valid_gt_dict.keys()))
        ds = TensorDataset(torch.from_numpy(val_users).long())
        return PyTorchDataLoader(ds, batch_size=batch_size, shuffle=False)

    def get_final_loader(self, batch_size):
        # Only users who have test data
        test_users = np.array(list(self.test_gt_dict.keys()))
        ds = TensorDataset(torch.from_numpy(test_users).long())
        return PyTorchDataLoader(ds, batch_size=batch_size, shuffle=False)

class RecSysDataset(Dataset):
    def __init__(self, users, items, n_items, user_history, loss_type, num_negatives, sampling_weights=None):
        self.users, self.items = users, items
        self.n_items, self.user_history = n_items, user_history
        self.loss_type, self.num_negatives = loss_type, num_negatives
        self.sampling_weights = sampling_weights

    def __len__(self): return len(self.users)

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
            seen = set(self.user_history.get(int(u_np[idx]), []))
            valid = [c for c in all_neg[idx] if c != int(i_np[idx]) and c not in seen]
            if len(valid) >= N: final_neg[idx] = valid[:N]
            else:
                final_neg[idx, :len(valid)] = valid
                for j in range(len(valid), N):
                    while True:
                        s = np.random.randint(0, self.n_items)
                        if s not in seen and s != int(i_np[idx]):
                            final_neg[idx, j] = s
                            break
        return {'user_id': torch.tensor(u_np, dtype=torch.long).unsqueeze(1), 'pos_item_id': torch.tensor(i_np, dtype=torch.long).unsqueeze(1), 'neg_item_id': torch.tensor(final_neg, dtype=torch.long)}
