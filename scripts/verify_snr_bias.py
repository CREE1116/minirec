import torch
import numpy as np
import pandas as pd
from src.data.loader import DataLoader
from src.models.ease import EASE
from src.models.aspire import Aspire
import matplotlib.pyplot as plt


def run_snr_bias_experiment(dataset_name='ml-100k'):
    # 1. 환경 설정 및 데이터 로드
    cfg = {
        'dataset_name': 'ml-100k',
        'data_path': 'data/ml100k/u.data',
        'separator': '\t',
        'columns': ['user_id', 'item_id', 'rating', 'timestamp'],
        'has_header': False,
        'min_user_interactions': 5,
        'min_item_interactions': 5,
        'rating_threshold': 4.0,
        'split_method': 'temporal_rs',
        'train_ratio': 0.8,
        'valid_ratio': 0.1,
        'seed': 42,
        'model': {'reg_lambda': 100.0, 'alpha': 1.0},
        'device': 'auto'
    }
    dl = DataLoader(cfg)
    n_users, n_items = dl.n_users, dl.n_items
    
    # R_true 생성
    full_df = pd.concat([dl.train_df, dl.valid_df, dl.test_df])
    rows = torch.tensor(full_df['user_id'].values)
    cols = torch.tensor(full_df['item_id'].values)
    R_true = torch.zeros((n_users, n_items))
    R_true[rows, cols] = 1.0
    
    print(f"Dataset: {dataset_name} | Users: {n_users}, Items: {n_items}")

    # ── STEP 1. Truly MCAR Masking ──
    p_obs = 0.2
    mask_mcar = (torch.rand((n_users, n_items)) < p_obs).float()
    R_obs = R_true * mask_mcar
    
    # ── STEP 2. SNR 측정 및 상관관계 분석 ──
    G = R_obs.t() @ R_obs
    d = G.diagonal()
    S = G.sum(dim=1)
    eps = 1e-12
    
    snr = (d**2) / (S + eps)
    log_snr = torch.log(snr + eps)
    snr_log_norm = torch.exp(log_snr - log_snr.mean())
    
    # SNR vs Popularity 상관관계
    item_pop = d.numpy()
    snr_vals = snr_log_norm.numpy()
    correlation = np.corrcoef(snr_vals, item_pop)[0, 1]
    print(f"\n[Step 2] SNR vs Popularity Correlation: {correlation:.4f}")
    if abs(correlation) > 0.8:
        print(" -> Warning: SNR is highly correlated with Popularity.")
    else:
        print(" -> SNR provides a relatively independent signal from Popularity.")

    # ── STEP 3. SNR 구간 분류 ──
    p33 = np.percentile(snr_vals, 33.3)
    p66 = np.percentile(snr_vals, 66.6)
    groups = {
        'Low SNR':  np.where(snr_vals < p33)[0],
        'Mid SNR':  np.where((snr_vals >= p33) & (snr_vals < p66))[0],
        'High SNR': np.where(snr_vals >= p66)[0]
    }

    # ── STEP 4. 구간별 관측 편향 분석 ──
    print("\n[Step 4] Observation Bias Analysis:")
    for name, items in groups.items():
        if len(items) == 0: continue
        true_mean = R_true[:, items].mean(dim=0)
        obs_count = mask_mcar[:, items].sum(dim=0)
        obs_sum = R_obs[:, items].sum(dim=0)
        valid = obs_count > 0
        bias = torch.abs(obs_sum[valid]/obs_count[valid] - true_mean[valid]).mean().item()
        print(f" {name:10s}: Bias={bias:.4f}, ItemCount={len(items)}")

    # ── STEP 5. Alpha Sweep (Inverting Alpha Test) ──
    print("\n[Step 5] Performance Sweep (RMSE) across Alpha values:")
    
    class MockDataLoader:
        def __init__(self, R):
            self.n_users, self.n_items = R.shape
            indices = R.nonzero().t()
            self.train_mat = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1]), R.shape).coalesce()
            self.train_df = pd.DataFrame({'user_id': indices[0].numpy(), 'item_id': indices[1].numpy()})
        def get_train_matrix(self, loader, dtype=torch.float32): return self.train_mat

    mock_dl = MockDataLoader(R_obs)
    
    # EASE (Baseline)
    ease = EASE(cfg, dl)
    ease.device = torch.device('cpu')
    ease.fit(mock_dl)
    R_pred_ease = (R_obs @ ease.weight_matrix)
    
    results = []
    # EASE 결과 저장
    for g_name, items in groups.items():
        rmse = torch.sqrt(torch.mean((R_pred_ease[:, items] - R_true[:, items])**2)).item()
        results.append({'Model': 'EASE', 'Alpha': 0.0, 'Group': g_name, 'RMSE': rmse})

    # Aspire Sweep: -1.0 (Invert), 0.5, 1.0, 2.0
    for alpha in [-1.0, 0.5, 1.0, 2.0]:
        cfg['model']['alpha'] = alpha
        aspire = Aspire(cfg, dl)
        aspire.device = torch.device('cpu')
        aspire.fit(mock_dl)
        R_pred = (R_obs @ aspire.weight_matrix)
        
        for g_name, items in groups.items():
            rmse = torch.sqrt(torch.mean((R_pred[:, items] - R_true[:, items])**2)).item()
            results.append({'Model': 'Aspire', 'Alpha': alpha, 'Group': g_name, 'RMSE': rmse})

    df_res = pd.DataFrame(results)
    pivot = df_res.pivot_table(index=['Model', 'Alpha'], columns='Group', values='RMSE')
    print("\n[Summary Table] RMSE Comparison (Lower is better)")
    print(pivot)

if __name__ == "__main__":
    run_snr_bias_experiment('ml-100k')
