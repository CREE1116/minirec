
import os
import torch
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import json
from src.utils.config import load_yaml
from src.data.loader import DataLoader

def get_matrix_from_df(df, n_users, n_items):
    rows = df['user_id'].values
    cols = df['item_id'].values
    vals = np.ones(len(rows))
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

def compute_metrics(evals, matrix):
    p = evals / (np.sum(evals) + 1e-12)
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    eff_rank = np.exp(entropy)
    
    row_sums = np.sum(np.abs(matrix), axis=1)
    row_sum_cv = np.std(row_sums) / (np.mean(row_sums) + 1e-12)
    
    return eff_rank, row_sum_cv, row_sums

def evaluate_recall(G, train_matrix, test_matrix, topk=20, reg=100.0):
    G_reg = G.copy()
    G_reg[np.diag_indices_from(G_reg)] += reg
    try:
        P = np.linalg.inv(G_reg)
    except np.linalg.LinAlgError:
        return 0.0
    W = -P / (np.diag(P)[np.newaxis, :] + 1e-12)
    np.fill_diagonal(W, 0)
    scores = train_matrix @ W 
    scores[train_matrix > 0] = -np.inf
    topk_indices = np.argsort(scores, axis=1)[:, -topk:]
    recalls = []
    for u in range(train_matrix.shape[0]):
        gt_items = test_matrix[u].nonzero()[1]
        if len(gt_items) == 0: continue
        hit = len(set(topk_indices[u]) & set(gt_items))
        recalls.append(hit / len(gt_items))
    return np.mean(recalls)

def plot_enhanced_comparison(results, output_path):
    n_methods = len(results)
    fig = plt.figure(figsize=(6 * n_methods, 18))
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
    
    baseline_name = 'Symmetric'
    baseline_evals = results[baseline_name]['evals']
    
    for i, (name, data) in enumerate(results.items()):
        ax1 = plt.subplot(3, n_methods, i + 1)
        subset_size = min(200, data['matrix'].shape[0])
        sub_matrix = np.abs(data['matrix'][:subset_size, :subset_size])
        vmax = np.percentile(sub_matrix[~np.eye(subset_size, dtype=bool)], 99)
        sns.heatmap(sub_matrix, ax=ax1, cmap='viridis', cbar=False, vmax=vmax)
        ax1.set_title(f"{name}\nR@20: {data['recall']:.4f}\nER: {data['eff_rank']:.1f}, CV: {data['row_sum_cv']:.3f}", 
                      fontsize=10, fontweight='bold')
        
        ax2 = plt.subplot(3, 1, 2)
        min_len = min(1000, len(data['evals']), len(baseline_evals))
        ratio = data['evals'][:min_len] / (baseline_evals[:min_len] + 1e-12)
        ax2.plot(ratio, label=f"{name} (R@20: {data['recall']:.4f})", color=colors[i], linewidth=2.5, alpha=0.8)
        
        ax3 = plt.subplot(3, 1, 3)
        rel_signal = data['row_sums'] / (np.mean(data['row_sums']) + 1e-12)
        sns.kdeplot(rel_signal, ax=ax3, label=name, color=colors[i], fill=True, alpha=0.2)

    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title("Information Preservation Ratio vs. Recall Performance", fontsize=16)
    ax2.set_yscale('log')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, which='both', alpha=0.3)

    ax3.set_title("Relative Item Signal Distribution (Normalized by Mean)", fontsize=16)
    ax3.set_xlim(0, 3)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Comparison plot saved to {output_path}")

def run_experiment():
    dataset_config = load_yaml("configs/datasets/ml-100k.yaml")
    data_loader = DataLoader(dataset_config)
    train_matrix = get_matrix_from_df(data_loader.train_df, data_loader.n_users, data_loader.n_items)
    test_matrix = get_matrix_from_df(data_loader.test_df, data_loader.n_users, data_loader.n_items)
    X = train_matrix
    X_dense = X.toarray()
    eps = 1e-12
    n_u = np.asarray(X.sum(axis=1)).ravel()
    d_i = np.asarray(X.sum(axis=0)).ravel()
    D_inv_half = sp.diags(np.power(d_i + eps, -0.5))
    
    results = {}
    
    def get_sym(): return (D_inv_half @ (X.T @ X) @ D_inv_half).toarray()
    def get_user_item(): return (D_inv_half @ (X.T @ sp.diags(np.power(n_u + eps, -1.0)) @ X) @ D_inv_half).toarray()
    def get_aspire(g):
        D_U_vst = sp.diags(np.power(n_u + eps, -g))
        X_vst = D_U_vst @ X
        item_bias = np.asarray(X_vst.power(2).sum(axis=0)).ravel()
        D_I_vst = sp.diags(np.power(item_bias + eps, -g/2.0))
        return (D_I_vst @ (X.T @ X_vst).toarray() @ D_I_vst)

    methods = {
        'Symmetric': get_sym,
        'User + Item': get_user_item,
        'ASPIRE (g=0.5)': lambda: get_aspire(0.5),
        'ASPIRE (g=1.0)': lambda: get_aspire(1.0)
    }
    
    for name, func in methods.items():
        print(f"Processing {name}...")
        matrix = func()
        matrix = (matrix + matrix.T) / 2.0
        evals = np.sort(np.linalg.eigvalsh(matrix))[::-1]
        eff_rank, row_sum_cv, row_sums = compute_metrics(evals, matrix)
        print(f"  Evaluating {name} performance...")
        recall = evaluate_recall(matrix, X_dense, test_matrix)
        results[name] = {
            'matrix': matrix, 'evals': evals, 'eff_rank': eff_rank, 
            'row_sum_cv': row_sum_cv, 'row_sums': row_sums, 'recall': recall
        }

    os.makedirs("output", exist_ok=True)
    plot_enhanced_comparison(results, "output/gram_proof_ml100k.png")

    json_results = {}
    for name, data in results.items():
        json_results[name] = {
            'effective_rank': float(data['eff_rank']),
            'row_sum_cv': float(data['row_sum_cv']),
            'recall_at_20': float(data['recall'])
        }
    with open("output/gram_comparison_results.json", 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=4, ensure_ascii=False)
    print("Metrics saved to output/gram_comparison_results.json")

if __name__ == "__main__":
    run_experiment()
