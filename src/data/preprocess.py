import os
import pandas as pd
import numpy as np
import argparse
import yaml
from typing import Dict, Any

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

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
    if separator == '::': engine = 'python'
    elif len(separator) > 1 and separator != r'\s+': engine = 'python'
    
    kwargs = {'sep': separator, 'engine': engine}
    if has_header:
        df = pd.read_csv(data_path, header=0, **kwargs)
        df = df[df.columns[:len(columns)]]
        df.columns = columns
    else:
        df = pd.read_csv(data_path, header=None, names=columns, **kwargs)
    return df

def filter_interactions(df, min_u, min_i, rating_threshold=0.0):
    if 'rating' in df.columns:
        df = df[df['rating'] >= rating_threshold].copy()
    
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
    """Remap user and item IDs to contiguous integers 0...N-1"""
    u_list = sorted(df['user_id'].unique())
    i_list = sorted(df['item_id'].unique())
    
    user_map = {old: new for new, old in enumerate(u_list)}
    item_map = {old: new for new, old in enumerate(i_list)}
    
    df['user_id'] = df['user_id'].map(user_map)
    df['item_id'] = df['item_id'].map(item_map)
    
    print(f"  Remapped: {len(user_map)} users, {len(item_map)} items")
    return df, user_map, item_map

def split_data(df, method='random', train_ratio=0.8, valid_ratio=0.1, seed=42):
    if len(df) == 0: return df.copy(), df.copy(), df.copy()
    
    if method == 'temporal_rs' and 'timestamp' in df.columns:
        df_sorted = df.sort_values(by=['user_id', 'timestamp'], kind='stable').reset_index(drop=True)
    else:
        df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        df_sorted = df_shuffled.sort_values(by='user_id', kind='stable').copy()
    
    total_counts = df_sorted.groupby('user_id', sort=False).size()
    cum_counts = df_sorted.groupby('user_id', sort=False).cumcount().values
    total_counts_v = total_counts.loc[df_sorted['user_id']].values
    
    train_end = np.clip((total_counts_v * train_ratio).astype(int), 1, None)
    valid_end = (total_counts_v * (train_ratio + valid_ratio)).astype(int)
    
    mask_lt3 = total_counts_v < 3
    if valid_ratio > 0:
        valid_end = np.where((total_counts_v >= 3) & (valid_end >= total_counts_v), total_counts_v - 1, valid_end)
        train_end = np.where((total_counts_v >= 3) & (train_end >= valid_end), valid_end - 1, train_end)
    else:
        valid_end = train_end
        train_end = np.where((total_counts_v >= 3) & (train_end >= total_counts_v), total_counts_v - 1, train_end)
    
    train = df_sorted[(cum_counts < train_end) | mask_lt3].copy()
    valid = df_sorted[(~mask_lt3) & (cum_counts >= train_end) & (cum_counts < valid_end)].copy()
    test = df_sorted[(~mask_lt3) & (cum_counts >= valid_end)].copy()
    
    return train, valid, test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to dataset yaml config')
    parser.add_argument('--output_dir', type=str, default='./data/preprocessed', help='Directory to save preprocessed datasets')
    args = parser.parse_args()

    if not args.config:
        print("Error: --config is required.")
        return

    config = load_yaml(args.config)
    dataset_name = config.get('dataset_name')
    out_path = os.path.join(args.output_dir, dataset_name)
    os.makedirs(out_path, exist_ok=True)

    print(f"Processing dataset: {dataset_name}")
    
    sep = config.get('separator', ',')
    columns = config.get('columns', ['user_id', 'item_id', 'rating', 'timestamp'])
    header = config.get('has_header', False)
    fmt = config.get('format', 'csv')
    min_u = config.get('min_user_interactions', 5)
    min_i = config.get('min_item_interactions', 5)
    rating_th = config.get('rating_threshold', 0.0)
    dedup = config.get('dedup', True)
    seed = config.get('seed', 42)

    if 'train_path' in config and 'test_path' in config:
        # Already split datasets
        train_raw = load_raw_data(config['train_path'], sep, columns, header, fmt)
        test_raw = load_raw_data(config['test_path'], sep, columns, header, fmt)
        full_raw = pd.concat([train_raw, test_raw])
    else:
        # Single file datasets
        data_path = config.get('data_path')
        full_raw = load_raw_data(data_path, sep, columns, header, fmt)

    if dedup:
        full_raw = full_raw.drop_duplicates(subset=['user_id', 'item_id'])
    
    # 1. Iterative Filter
    df_filtered = filter_interactions(full_raw, min_u, min_i, rating_th)
    
    # 2. Contiguous Remap (0 to N-1)
    df_remapped, _, _ = remap_ids(df_filtered)
    
    # 3. Split
    split_method = config.get('split_method', 'random')
    train_ratio = config.get('train_ratio', 0.8)
    valid_ratio = config.get('valid_ratio', 0.1)
    
    train, valid, test = split_data(df_remapped, split_method, train_ratio, valid_ratio, seed)

    # 4. Save
    train.to_csv(os.path.join(out_path, 'train.csv'), index=False)
    valid.to_csv(os.path.join(out_path, 'valid.csv'), index=False)
    test.to_csv(os.path.join(out_path, 'test.csv'), index=False)
    
    print(f"Dataset {dataset_name} processed: {len(train)} train, {len(valid)} valid, {len(test)} test")
    print(f"Final Count: {df_remapped['user_id'].nunique()} users, {df_remapped['item_id'].nunique()} items")
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
