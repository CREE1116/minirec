import os
import pandas as pd
import numpy as np
import argparse
import yaml
import gc
from typing import Dict, Any, Tuple

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def load_raw_data(data_path: str, separator: str, columns: list, has_header: bool = False, fmt: str = 'csv') -> pd.DataFrame:
    if fmt == 'lightgcn':
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
    
    engine = 'python' if len(separator) > 1 else 'c'
    kwargs = {'sep': separator, 'engine': engine, 'header': 0 if has_header else None}
    
    df = pd.read_csv(data_path, **kwargs)
    if not has_header:
        df.columns = columns[:len(df.columns)]
    else:
        df = df[df.columns[:len(columns)]]
        df.columns = columns
    return df

def filter_interactions(df: pd.DataFrame, min_u: int, min_i: int) -> pd.DataFrame:
    """Iterative k-core filtering."""
    curr_len = len(df)
    while True:
        u_counts = df.groupby('user_id').size()
        df = df[df['user_id'].isin(u_counts[u_counts >= min_u].index)]
        
        i_counts = df.groupby('item_id').size()
        df = df[df['item_id'].isin(i_counts[i_counts >= min_i].index)]
        
        new_len = len(df)
        if curr_len == new_len:
            break
        curr_len = new_len
    return df

def filter_train_core(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, min_u: int, min_i: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ensure k-core constraint is met strictly on the TRAINING set."""
    print(f"[Preprocess] Starting strict train-core filtering (min_u={min_u}, min_i={min_i})...")
    while True:
        start_users = train['user_id'].nunique()
        start_items = train['item_id'].nunique()
        
        # 1. Filter items based on Train set counts
        i_counts = train['item_id'].value_counts()
        valid_items = i_counts[i_counts >= min_i].index
        
        train = train[train['item_id'].isin(valid_items)]
        valid = valid[valid['item_id'].isin(valid_items)]
        test = test[test['item_id'].isin(valid_items)]
        
        # 2. Filter users based on Train set counts
        u_counts = train['user_id'].value_counts()
        valid_users = u_counts[u_counts >= min_u].index
        
        train = train[train['user_id'].isin(valid_users)]
        valid = valid[valid['user_id'].isin(valid_users)]
        test = test[test['user_id'].isin(valid_users)]
        
        if train['user_id'].nunique() == start_users and train['item_id'].nunique() == start_items:
            break
            
    print(f"[Preprocess] Final Train-core: {train['user_id'].nunique()} users, {train['item_id'].nunique()} items.")
    return train, valid, test

def remap_ids(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict]:
    """Fast remapping using pandas Categorical."""
    df = df.copy()
    df['user_id'] = pd.Categorical(df['user_id'])
    df['item_id'] = pd.Categorical(df['item_id'])
    
    user_map = {old: new for new, old in enumerate(df['user_id'].cat.categories)}
    item_map = {old: new for new, old in enumerate(df['item_id'].cat.categories)}
    
    df['user_id'] = df['user_id'].cat.codes
    df['item_id'] = df['item_id'].cat.codes
    
    return df, user_map, item_map

def split_data(df: pd.DataFrame, method: str = 'random', train_ratio: float = 0.8, valid_ratio: float = 0.1, seed: int = 42):
    """Split data and handle cold-start items in eval sets."""
    if method == 'temporal_rs' and 'timestamp' in df.columns:
        df = df.sort_values(by=['user_id', 'timestamp'], kind='stable')
    else:
        df = df.sample(frac=1, random_state=seed).sort_values(by='user_id', kind='stable')

    group = df.groupby('user_id', sort=False)
    cum_counts = group.cumcount()
    total_counts = group.transform('size')

    # Ensure at least 1 item in train if user passes k-core
    train_end = (total_counts * train_ratio).astype(int).clip(lower=1)
    valid_end = (total_counts * (train_ratio + valid_ratio)).astype(int)

    train = df[cum_counts < train_end].copy()
    valid = df[(cum_counts >= train_end) & (cum_counts < valid_end)].copy()
    test = df[cum_counts >= valid_end].copy()

    return train, valid, test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./data/preprocessed')
    args = parser.parse_args()

    config = load_yaml(args.config)
    dataset_name = config.get('dataset_name', 'unknown')
    out_path = os.path.join(args.output_dir, dataset_name)
    os.makedirs(out_path, exist_ok=True)

    min_u = config.get('min_user_interactions', 5)
    min_i = config.get('min_item_interactions', 5)

    # 1. Load Data
    if 'train_path' in config and 'test_path' in config:
        train_raw = load_raw_data(config['train_path'], config.get('separator', ','), config['columns'], config.get('has_header', False), config.get('format', 'csv'))
        test_raw = load_raw_data(config['test_path'], config.get('separator', ','), config['columns'], config.get('has_header', False), config.get('format', 'csv'))
        full_raw = pd.concat([train_raw, test_raw])
    else:
        full_raw = load_raw_data(config['data_path'], config.get('separator', ','), config['columns'], config.get('has_header', False), config.get('format', 'csv'))

    # 2. Deduplication
    if config.get('dedup', True):
        sort_cols = []
        if 'timestamp' in full_raw.columns: sort_cols.append('timestamp')
        if 'rating' in full_raw.columns: sort_cols.append('rating')
        if sort_cols:
            full_raw = full_raw.sort_values(by=sort_cols, ascending=False)
        full_raw = full_raw.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    
    # 3. Rating Threshold
    if 'rating' in full_raw.columns:
        full_raw = full_raw[full_raw['rating'] >= config.get('rating_threshold', 0.0)]

    # 4. Global K-core filtering (Preliminary)
    df_filtered = filter_interactions(full_raw, min_u, min_i)
    
    # 5. Split Data
    train, valid, test = split_data(
        df_filtered, 
        method=config.get('split_method', 'random'),
        train_ratio=config.get('train_ratio', 0.8),
        valid_ratio=config.get('valid_ratio', 0.1),
        seed=config.get('seed', 42)
    )

    # 6. Strict Train Core Filtering (The Fix)
    # Relax the threshold for the TRAIN set specifically (e.g., 10 total -> 8 in train)
    train_ratio = config.get('train_ratio', 0.8)
    eff_min_u = max(1, int(min_u * train_ratio))
    eff_min_i = max(1, int(min_i * train_ratio))
    
    print(f"[Preprocess] Training set threshold: users >= {eff_min_u}, items >= {eff_min_i} (based on {train_ratio*100}% split)")
    train, valid, test = filter_train_core(train, valid, test, eff_min_u, eff_min_i)
    
    # 7. Remap IDs (Must be done at the very end to ensure contiguous IDs)
    len_train = len(train)
    len_valid = len(valid)
    full_final = pd.concat([train, valid, test])
    
    df_remapped, user_map, item_map = remap_ids(full_final)
    
    train = df_remapped.iloc[:len_train].copy()
    valid = df_remapped.iloc[len_train : len_train + len_valid].copy()
    test = df_remapped.iloc[len_train + len_valid:].copy()

    # 8. Save
    train.to_csv(os.path.join(out_path, 'train.csv'), index=False)
    valid.to_csv(os.path.join(out_path, 'valid.csv'), index=False)
    test.to_csv(os.path.join(out_path, 'test.csv'), index=False)
    
    print(f"Preprocessing for {dataset_name} complete. IDs remapped and train-core verified.")

if __name__ == '__main__':
    main()
