
import os
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import json
from src.utils.config import load_yaml
from src.data.loader import DataLoader

def run_spectral_analysis():
    # 1. 데이터 로드
    dataset_config = load_yaml("configs/datasets/ml-100k.yaml")
    data_loader = DataLoader(dataset_config)
    
    rows = data_loader.train_df['user_id'].values
    cols = data_loader.train_df['item_id'].values
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), 
                     shape=(data_loader.n_users, data_loader.n_items))
    
    eps = 1e-12
    n_u = np.asarray(X.sum(axis=1)).ravel()
    d_i = np.asarray(X.sum(axis=0)).ravel()
    
    # --- 행렬 계산 ---
    print("Computing Normalizations...")
    
    # A. Raw Gram
    G_raw = (X.T @ X).toarray()
    
    # B. Symmetric Normalization
    D_inv_half = sp.diags(np.power(d_i + eps, -0.5))
    G_sym = (D_inv_half @ (X.T @ X) @ D_inv_half).toarray()
    
    # C. ASPIRE (Variance Stabilizing) 함수
    def get_aspire(g):
        D_U_vst = sp.diags(np.power(n_u + eps, -g))
        X_vst = D_U_vst @ X
        item_bias = np.asarray(X_vst.power(2).sum(axis=0)).ravel()
        D_I_vst = sp.diags(np.power(item_bias + eps, -g/2.0))
        return (D_I_vst @ (X.T @ X_vst).toarray() @ D_I_vst)

    matrices = {
        'Raw': G_raw, 
        'Symmetric': G_sym, 
        'ASPIRE (g=0.5)': get_aspire(0.5),
        'ASPIRE (g=1.0)': get_aspire(1.0)
    }
    
    # 시각화 설정
    n_methods = len(matrices)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
    color_dict = {name: colors[i] for i, name in enumerate(matrices.keys())}
    
    # --- 1. 대각 성분 분포 ---
    ax1 = axes[0, 0]
    for name, G in matrices.items():
        if name == 'Raw': continue 
        sns.kdeplot(np.diag(G), ax=ax1, label=name, color=color_dict[name], fill=True, alpha=0.2)
    
    ax1.set_title("Variance Stabilization: Diagonal Element Distribution", fontsize=14)
    ax1.set_xlabel("Value of Diagonal Elements ($G_{ii}$)")
    ax1.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
    ax1.legend()

    # --- 2. 고윳값 분포 ---
    ax2 = axes[0, 1]
    results = {}
    for name, G in matrices.items():
        print(f"Analyzing Spectrum of {name}...")
        G_fixed = (G + G.T) / 2.0
        evals = np.sort(np.linalg.eigvalsh(G_fixed))[::-1]
        evals_norm = evals / (evals[0] + eps)
        results[name] = {'evals': evals, 'evals_norm': evals_norm}
        ax2.plot(evals_norm[:500], label=name, color=color_dict[name], linewidth=2)
    
    ax2.set_title("Restore Isotropy: Eigenvalue Decay Rate", fontsize=14)
    ax2.set_yscale('log')
    ax2.set_ylabel("Normalized Eigenvalue (Log)")
    ax2.grid(True, which='both', linestyle='--', alpha=0.3)
    ax2.legend()

    # --- 3. 누적 에너지 ---
    ax3 = axes[1, 0]
    for name, data in results.items():
        cum_energy = np.cumsum(data['evals']) / np.sum(data['evals'])
        ax3.plot(cum_energy[:500], label=name, color=color_dict[name], linewidth=2)
    
    ax3.set_title("Energy Concentration: Cumulative Energy Ratio", fontsize=14)
    ax3.set_ylabel("Cumulative Energy")
    ax3.axhline(y=0.9, color='green', linestyle=':', label='90% Energy')
    ax3.legend()

    # --- 4. Condition Number ---
    ax4 = axes[1, 1]
    names, cond_nums = [], []
    for name, data in results.items():
        pos_evals = data['evals'][data['evals'] > 1e-5]
        cond = pos_evals[0] / pos_evals[-1]
        names.append(name)
        cond_nums.append(cond)
    
    ax4.bar(names, cond_nums, color=[color_dict[n] for n in names])
    ax4.set_yscale('log')
    ax4.set_title("Condition Number (Lower is More Stable)", fontsize=14)
    plt.xticks(rotation=15)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/spectral_analysis_ml100k.png", dpi=150)
    
    # --- 5. 수치 데이터 JSON 저장 ---
    spectral_metrics = {}
    for name, G in matrices.items():
        diags = np.diag(G)
        evals = results[name]['evals']
        pos_evals = evals[evals > 1e-5]
        
        spectral_metrics[name] = {
            "condition_number": float(pos_evals[0] / pos_evals[-1]),
            "diagonal_mean": float(np.mean(diags)),
            "diagonal_std": float(np.std(diags)),
            "diagonal_cv": float(np.std(diags) / (np.mean(diags) + eps)),
            "energy_90_rank": int(np.searchsorted(np.cumsum(evals) / np.sum(evals), 0.9)),
            "top_10_energy_ratio": float(np.sum(evals[:10]) / np.sum(evals))
        }

    with open("output/spectral_analysis_results.json", 'w', encoding='utf-8') as f:
        json.dump(spectral_metrics, f, indent=4, ensure_ascii=False)
    print("Spectral Analysis complete. Metrics saved to output/spectral_analysis_results.json")

if __name__ == "__main__":
    run_spectral_analysis()
