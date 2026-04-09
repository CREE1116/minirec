import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from src.utils.config import load_yaml
from src.data.loader import DataLoader
from src.models.base import BaseModel

def get_ace_distribution(dataset_cfg_path):
    cfg = load_yaml(dataset_cfg_path)
    dl = DataLoader(cfg)
    dummy_model = BaseModel(cfg, dl)
    X = dummy_model.get_train_matrix(dl).to(dummy_model.device)
    G = torch.sparse.mm(X.t(), X.to_dense())
    
    eps = 1e-12
    d = G.diagonal()
    S = G.sum(dim=1)
    log_ace = 2.0 * torch.log(d + eps) - torch.log(S + eps)
    ace = torch.exp(log_ace - log_ace.mean())
    return ace.cpu().numpy()

def plot_mnar_analysis():
    os.makedirs('output/analysis', exist_ok=True)
    
    # --- 데이터 준비 (사용자 제공 지표 기반) ---
    datasets = ['ML-100k', 'ML-1M', 'Steam']
    corrs = [0.83, 0.91, 0.95]
    alphas = [1.0, 0.7, 0.2]
    kurtosis_vals = [16.6, 28.6, 375.8]
    ndcg_gain = [0.012, 0.008, 0.002] 
    
    # 1. MNAR 강도 및 보정 효과 통합 그래프
    plt.figure(figsize=(8, 6))
    ax1 = plt.gca()
    
    # Alpha* (Left Axis)
    color1 = 'tab:blue'
    ax1.set_xlabel('corr(ACE, S_i)', fontsize=12)
    ax1.set_ylabel('Optimal Alpha*', color=color1, fontsize=12)
    ax1.plot(corrs, alphas, 'o-', color=color1, linewidth=3, markersize=10, label='Optimal Alpha*')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # NDCG Gain (Right Axis)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('NDCG Improvement (Aspire - EASE)', color=color2, fontsize=12)
    ax2.plot(corrs, ndcg_gain, 's--', color=color2, linewidth=3, markersize=10, label='NDCG Gain')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    for i, txt in enumerate(datasets):
        ax1.text(corrs[i], alphas[i]+0.02, txt, fontweight='bold', ha='center')

    plt.title('MNAR Intensity vs Correction Strength & Effect', fontsize=14)
    plt.tight_layout()
    plt.savefig('output/analysis/mnar_correlation_impact.png')
    plt.close()
    print("Saved: output/analysis/mnar_correlation_impact.png")

    # 2. ACE Kurtosis
    plt.figure(figsize=(7, 5))
    x_pos = np.arange(len(datasets))
    plt.bar(x_pos, kurtosis_vals, color=['skyblue', 'steelblue', 'darkred'], alpha=0.8)
    plt.xticks(x_pos, datasets)
    plt.yscale('log')
    plt.ylabel('ACE Kurtosis (log scale)', fontsize=12)
    plt.title('ACE Distribution Explosion (Kurtosis)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(kurtosis_vals):
        plt.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/analysis/ace_kurtosis.png')
    plt.close()
    print("Saved: output/analysis/ace_kurtosis.png")

    # 3. ACE Distribution Overlay (핵심: Log Scale KDE 스타일)
    print("\n>>> Calculating real ACE distributions for Overlay Plot...")
    configs = [
        'configs/datasets/ml-100k.yaml',
        'configs/datasets/ml-1m.yaml',
        'configs/datasets/steam.yaml'
    ]
    
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red']
    
    for i, cfg_path in enumerate(configs):
        d_name = datasets[i]
        try:
            ace_vals = get_ace_distribution(cfg_path)
            # Log scale histogram (Density plot style)
            log_ace = np.log10(ace_vals + 1e-12)
            counts, bin_edges = np.histogram(log_ace, bins=50, density=True)
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            plt.plot(bin_centres, counts, color=colors[i], linewidth=2.5, label=d_name)
            plt.fill_between(bin_centres, counts, color=colors[i], alpha=0.1)
            
        except Exception as e:
            print(f"Failed to load {d_name}: {e}")

    plt.title('Distribution of ACE values (Log10 Scale)', fontsize=14)
    plt.xlabel('Log10(ACE Value)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('output/analysis/ace_distribution_overlay.png')
    plt.close()
    print("Saved: output/analysis/ace_distribution_overlay.png")

if __name__ == "__main__":
    plot_mnar_analysis()
