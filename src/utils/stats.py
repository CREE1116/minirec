import os
import pandas as pd
import numpy as np

def get_dataset_stats(dataset_name, base_path=None):
    if base_path is None:
        base_path = os.path.join(os.getcwd(), 'data', 'preprocessed')
    
    data_dir = os.path.join(base_path, dataset_name)
    
    if not os.path.exists(data_dir):
        return None

    try:
        train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        valid_df = pd.read_csv(os.path.join(data_dir, 'valid.csv'))
        test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None

    full_df = pd.concat([train_df, valid_df, test_df])
    
    n_users = full_df['user_id'].nunique()
    n_items = full_df['item_id'].nunique()
    n_train = len(train_df)
    n_valid = len(valid_df)
    n_test = len(test_df)
    n_total = len(full_df)
    
    density = (n_total / (n_users * n_items)) * 100
    
    user_counts = train_df.groupby('user_id')['item_id'].count()
    item_counts = train_df.groupby('item_id')['user_id'].count()
    
    stats = {
        'Dataset': dataset_name,
        'Users': n_users,
        'Items': n_items,
        'Interactions': n_total,
        'Train/Valid/Test': f"{n_train}/{n_valid}/{n_test}",
        'Density (%)': f"{density:.4f}%",
        'Avg Inter/User': f"{user_counts.mean():.2f}",
        'Max Inter/User': user_counts.max(),
        'Min Inter/User': user_counts.min(),
        'Avg Inter/Item': f"{item_counts.mean():.2f}",
        'Max Inter/Item': item_counts.max(),
        'Min Inter/Item': item_counts.min(),
    }
    return stats
