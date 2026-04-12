import os
import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy import sparse

# Add root directory to path
sys.path.append(os.getcwd())

def calculate_gini(array):
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    if array.sum() == 0: return 0.0
    return ((2 * index - n - 1) * array).sum() / (n * array.sum() + 1e-12)

def calculate_effective_rank(eigvals):
    norm_eig = eigvals / (eigvals.sum() + 1e-12)
    entropy = -np.sum(norm_eig * np.log(norm_eig + 1e-12))
    return np.exp(entropy)

def generate_mnar_data(n_users=1000, n_items=1500, latent_dim=20, gamma=2.0):
    """
    Generate True Preference (MCAR) and Biased Observation (MNAR).
    """
    U = np.random.randn(n_users, latent_dim)
    V = np.random.randn(n_items, latent_dim)
    true_scores = U @ V.T
    true_matrix = (true_scores > np.percentile(true_scores, 95)).astype(float)
    
    item_ranks = np.arange(1, n_items + 1)
    intrinsic_popularity = 1.0 / np.power(item_ranks, 0.7)
    intrinsic_popularity /= intrinsic_popularity.max()
    
    exposure_prob = np.power(intrinsic_popularity, gamma)
    exposure_mask = np.random.rand(n_users, n_items) < exposure_prob[None, :]
    
    observed_matrix = true_matrix * exposure_mask
    return true_matrix, observed_matrix, intrinsic_popularity

def run_detailed_mnar_structural_analysis():
    print("Running Detailed MNAR Structural Analysis...")
    base_dir = 'output/analytic/mnar_structural_analysis'
    os.makedirs(base_dir, exist_ok=True)
    
    true_mat, obs_mat, exposure_probs = generate_mnar_data(gamma=2.0)
    
    true_item_pop = true_mat.sum(axis=0)
    obs_item_pop = obs_mat.sum(axis=0)
    
    # Gram matrices
    G_true = true_mat.T @ true_mat
    G_obs = obs_mat.T @ obs_mat
    
    def norm_gram(G):
        d = np.sqrt(np.diag(G) + 1e-9)
        return G / (d[:, None] * d[None, :])

    G_true_norm = norm_gram(G_true)
    G_obs_norm = norm_gram(G_obs)
    
    # 1. Plot: Popularity Distortion
    plt.figure(figsize=(10, 8))
    plt.scatter(true_item_pop, obs_item_pop, alpha=0.4, color='tab:blue')
    plt.title("Popularity Distortion: True Preference vs Observed Clicks", fontsize=14)
    plt.xlabel("True Preference Count (Unbiased)", fontsize=12)
    plt.ylabel("Observed Click Count (Biased)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{base_dir}/01_popularity_distortion.png", dpi=200)
    plt.close()

    # 2. Plot: Spectral Collapse (Eigenvalue Decay)
    plt.figure(figsize=(10, 8))
    e_true = np.sort(np.linalg.eigvalsh(G_true))[::-1]
    e_obs = np.sort(np.linalg.eigvalsh(G_obs))[::-1]
    plt.plot(e_true[:150] / e_true[0], label='True Structure (MCAR)', linestyle='--', color='gray', linewidth=2)
    plt.plot(e_obs[:150] / e_obs[0], label='Observed Structure (MNAR)', color='tab:orange', linewidth=3)
    plt.yscale('log')
    plt.title("Spectral Collapse: Information Loss via Bias", fontsize=14)
    plt.ylabel("Normalized Eigenvalue (Log Scale)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(f"{base_dir}/02_spectral_collapse.png", dpi=200)
    plt.close()

    # 3. Plot: Similarity Banding (Heatmap - Observed)
    plt.figure(figsize=(10, 8))
    plt.imshow(G_obs_norm[:100, :100], cmap='viridis')
    plt.title("Similarity Banding: Popularity 'Shadows' in Data", fontsize=14)
    plt.colorbar(label='Normalized Co-occurrence')
    plt.savefig(f"{base_dir}/03_similarity_banding_mnar.png", dpi=200)
    plt.close()

    # 4. Plot: SNR Distribution (Spectral Signal-to-Noise)
    plt.figure(figsize=(10, 8))
    snr_obs = np.log((np.diag(G_obs) + 1e-9) / (G_obs.sum(axis=1) + 1e-9))
    plt.scatter(np.log1p(obs_item_pop), snr_obs, alpha=0.5, color='tab:red', s=15)
    plt.axhline(y=snr_obs.mean(), color='black', linestyle='--')
    plt.title("Item SNR vs Popularity: The Target for ASPIRE", fontsize=14)
    plt.xlabel("Log(Observed Popularity)", fontsize=12)
    plt.ylabel("Spectral SNR (log diag/rowsum)", fontsize=12)
    plt.savefig(f"{base_dir}/04_item_snr_distribution.png", dpi=200)
    plt.close()

    # --- Gamma Sweep Section ---
    print("Running MNAR Gamma Sweep...")
    gammas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    sweep_data = []
    for g in gammas:
        t_m, o_m, _ = generate_mnar_data(gamma=g)
        o_pop = o_m.sum(axis=0)
        
        X = sparse.csr_matrix(o_m)
        if o_m.sum() > 0:
            G = X.T.dot(X).toarray()
            eigvals = np.sort(np.linalg.eigvalsh(G))[::-1]
            eigvals = np.maximum(eigvals, 0)
            er = calculate_effective_rank(eigvals)
        else:
            er = 0.0
            
        sweep_data.append({
            'gamma': g,
            'gini': calculate_gini(o_pop),
            'effective_rank': er,
            'retention': float(o_m.sum() / t_m.sum())
        })

    df_sweep = pd.DataFrame(sweep_data)
    
    # 5. Plot: Sweep Results (Gini & Effective Rank)
    plt.figure(figsize=(10, 8))
    plt.plot(df_sweep['gamma'], df_sweep['gini'], marker='s', label='Gini Index (Inequality)', color='tab:red', linewidth=2)
    plt.plot(df_sweep['gamma'], df_sweep['effective_rank'] / df_sweep['effective_rank'].max(), 
             marker='o', label='Normalized Effective Rank', color='tab:blue', linewidth=2)
    plt.title("Impact of Bias Strength (Gamma) on Data Health", fontsize=14)
    plt.xlabel("Exposure Bias Gamma", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{base_dir}/05_gamma_sweep_impact.png", dpi=200)
    plt.close()

    # Save final JSON
    with open(f"{base_dir}/mnar_analysis_summary.json", 'w') as f:
        json.dump(sweep_data, f, indent=4)
    
    print(f"✨ MNAR Structural Analysis complete! Plots saved in: {base_dir}")

if __name__ == "__main__":
    run_detailed_mnar_structural_analysis()
