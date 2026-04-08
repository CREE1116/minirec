import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.data.loader import DataLoader

def visualize_stats(dataset_name='ml-100k'):
    # 데이터 로드
    configs = {
        'ml-100k': {
            'dataset_name': 'ml-100k',
            'data_path': 'data/ml100k/u.data',
            'separator': '\t',
            'columns': ['user_id', 'item_id', 'rating', 'timestamp'],
            'has_header': False,
            'rating_threshold': 4.0
        },
        'ml-1m': {
            'dataset_name': 'ml-1m',
            'data_path': 'data/ml1m/ratings.dat',
            'separator': '::',
            'columns': ['user_id', 'item_id', 'rating', 'timestamp'],
            'has_header': False,
            'rating_threshold': 4.0
        },
        'steam': {
            'dataset_name': 'steam',
            'data_path': 'data/steam/ratings.dat',
            'separator': '::',
            'columns': ['user_id', 'item_id', 'rating', 'timestamp'],
            'has_header': False,
            'rating_threshold': 0.0
        }
    }
    
    if dataset_name not in configs:
        print(f"Dataset {dataset_name} not supported.")
        return

    c = configs[dataset_name]
    cfg = {
        'dataset_name': c['dataset_name'],
        'data_path': c['data_path'],
        'separator': c['separator'],
        'columns': c['columns'],
        'has_header': c['has_header'],
        'min_user_interactions': 5,
        'min_item_interactions': 5,
        'rating_threshold': c['rating_threshold'],
        'split_method': 'temporal_rs',
        'train_ratio': 0.8,
        'valid_ratio': 0.1,
        'seed': 42
    }
    
    dl = DataLoader(cfg)
    
    # R_true 생성 (전체 데이터)
    full_df = pd.concat([dl.train_df, dl.valid_df, dl.test_df])
    rows = torch.tensor(full_df['user_id'].values)
    cols = torch.tensor(full_df['item_id'].values)
    X = torch.zeros((dl.n_users, dl.n_items))
    X[rows, cols] = 1.0
    
    # ── 지표 계산 ──
    G = X.t() @ X
    d = G.diagonal()       # Degree (Popularity)
    S = G.sum(dim=1)       # Row Sum (Co-occurrence Energy)
    eps = 1e-12
    
    # Aspire SNR (Log-space)
    log_d = torch.log(d + eps)
    log_S = torch.log(S + eps)
    log_lambda = 2.0 * log_d - log_S
    log_lambda_centered = log_lambda - log_lambda.mean()
    snr = log_lambda_centered.exp()

    # 데이터프레임화
    stats = pd.DataFrame({
        'Degree (d)': d.numpy(),
        'Row Sum (S)': S.numpy(),
        'SNR (λ)': snr.numpy(),
        'Log SNR': log_lambda_centered.numpy()
    })

    # ── 시각화 ──
    plt.figure(figsize=(16, 10))

    # 1. Degree 분포
    plt.subplot(2, 3, 1)
    plt.hist(stats['Degree (d)'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'[{dataset_name}] Distribution of Degree (d)')
    plt.xlabel('Degree')
    plt.ylabel('Count')

    # 2. Row Sum 분포
    plt.subplot(2, 3, 2)
    plt.hist(stats['Row Sum (S)'], bins=30, color='salmon', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Row Sum (S)')
    plt.xlabel('Row Sum')

    # 3. SNR 분포
    plt.subplot(2, 3, 3)
    plt.hist(stats['SNR (λ)'], bins=30, color='green', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Aspire SNR (λ)')
    plt.xlabel('SNR')

    # 4. Degree vs Row Sum
    plt.subplot(2, 3, 4)
    plt.scatter(stats['Degree (d)'], stats['Row Sum (S)'], alpha=0.5, s=10)
    max_d = stats['Degree (d)'].max()
    plt.plot([0, max_d], [0, max_d], 'r--', label='d = S (Max SNR)')
    plt.title('Degree vs Row Sum')
    plt.xlabel('Degree (d)')
    plt.ylabel('Row Sum (S)')
    plt.legend()

    # 5. Degree vs SNR
    plt.subplot(2, 3, 5)
    plt.scatter(stats['Degree (d)'], stats['SNR (λ)'], alpha=0.5, s=10, color='green')
    plt.title('Degree vs SNR')
    plt.xlabel('Degree (d)')
    plt.ylabel('SNR (λ)')

    # 6. Row Sum vs Log SNR
    plt.subplot(2, 3, 6)
    plt.scatter(stats['Row Sum (S)'], stats['Log SNR'], alpha=0.5, s=10, color='purple')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Row Sum vs Log SNR')
    plt.xlabel('Row Sum (S)')
    plt.ylabel('Log SNR')

    plt.tight_layout()
    output_filename = f'aspire_stats_{dataset_name}.png'
    plt.savefig(output_filename)
    plt.close()
    print(f"Visualization saved as '{output_filename}'")
    
    # 수치 요약 출력
    print(f"\n[{dataset_name} Summary Statistics]")
    print(stats.describe())
    print(f"Correlation (d vs SNR): {stats['Degree (d)'].corr(stats['SNR (λ)']):.4f}\n")

if __name__ == "__main__":
    for ds in ['ml-100k', 'ml-1m', 'steam']:
        visualize_stats(ds)
