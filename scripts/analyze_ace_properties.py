import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy import sparse
from tqdm import tqdm

# Add root directory to path
sys.path.append(os.getcwd())

from src.models.ease import EASE
from src.models.causal_aspire import CausalAspire
from src.utils.sparse import _GLOBAL_SPARSE_CACHE

class HarsherSimulation:
    """Simulation with extreme bias and random interaction noise."""
    def __init__(self, n_users=1000, n_items=1500, latent_dim=20, noise_ratio=0.03):
        self.n_users = n_users
        self.n_items = n_items
        
        # 1. Base Truth
        U = np.random.randn(n_users, latent_dim)
        V = np.random.randn(n_items, latent_dim)
        V[0:100] *= 4.0 # High quality gems & stars
        V[100:150] *= 0.1 # Low quality junk
        
        true_scores = U @ V.T
        self.true_matrix = (true_scores > np.percentile(true_scores, 95)).astype(float)
        
        # 2. Extreme MNAR Exposure (Gamma=3.0)
        item_ranks = np.arange(1, n_items + 1)
        exposure_prob = 1.0 / np.power(item_ranks, 1.2) # Sharper decay
        exposure_prob /= exposure_prob.max()
        # High exposure for stars
        exposure_prob[50:150] = 0.9 
        exposure_prob[0:50] = 0.005 # Extremely hidden gems
        
        exposure_mask = np.random.rand(n_users, n_items) < exposure_prob[None, :]
        self.observed_signal = self.true_matrix * exposure_mask
        
        # 3. Inject Pure Random Noise (Spam Clicks)
        # noise_ratio of entries are randomly flipped to 1
        random_noise = (np.random.rand(n_users, n_items) < noise_ratio).astype(float)
        self.train_matrix = np.maximum(self.observed_signal, random_noise)
        
        self.test_matrix = self.true_matrix * (self.train_matrix == 0)
        
        self.groups = {
            'HiddenGems': list(range(0, 50)),
            'RealStars': list(range(50, 100)),
            'FakeStars': list(range(100, 150))
        }

    def get_loader(self):
        class MockLoader:
            def __init__(self, m, u, i):
                self.n_users, self.n_items = u, i
                r, c = m.nonzero()
                self.train_df = pd.DataFrame({'user_id': r, 'item_id': c})
                self.cache_filename = f"ace_sim_{np.random.randint(1e9)}"
        return MockLoader(self.train_matrix, self.n_users, self.n_items)

def calculate_effective_rank(eigvals):
    norm_eig = eigvals / (eigvals.sum() + 1e-12)
    entropy = -np.sum(norm_eig * np.log(norm_eig + 1e-12))
    return np.exp(entropy)

def run_ace_analysis():
    print("Starting ACE (Adaptive Causal Ensemble) Robustness Analysis...")
    base_dir = 'output/analytic/ace_properties_analysis'
    os.makedirs(base_dir, exist_ok=True)
    
    # Sweep noise levels
    noise_levels = [0.0, 0.02, 0.05, 0.10]
    alphas = [0.0, 0.5, 1.0, 1.5]
    
    all_results = []

    for n_lvl in noise_levels:
        print(f"\nTesting Noise Level: {n_lvl*100}%")
        sim = HarsherSimulation(noise_ratio=n_lvl)
        dl = sim.get_loader()
        
        for a in alphas:
            _GLOBAL_SPARSE_CACHE.clear()
            config = {'model': {'reg_lambda': 50.0, 'alpha': a, 'beta': 0.6}, 'device': 'cpu'}
            model = CausalAspire(config, dl)
            model.fit(dl)
            
            with torch.no_grad():
                scores = model.forward(torch.arange(sim.n_users)).numpy()
            scores[sim.train_matrix > 0] = -1e10
            
            # Metrics
            all_ranks = np.argsort(np.argsort(-scores, axis=1), axis=1)
            rs_rank = all_ranks[:, sim.groups['RealStars']].mean()
            fs_rank = all_ranks[:, sim.groups['FakeStars']].mean()
            hg_rank = all_ranks[:, sim.groups['HiddenGems']].mean()
            
            # Disentanglement: Gap between Fake and Real
            disentangle = fs_rank - rs_rank 
            
            # Spectral Info
            X = sparse.csr_matrix(sim.train_matrix)
            G = X.T.dot(X).toarray()
            eigvals = np.sort(np.linalg.eigvalsh(G))[::-1]
            er = calculate_effective_rank(np.maximum(eigvals, 0))
            
            all_results.append({
                'noise': n_lvl,
                'alpha': a,
                'rs_rank': float(rs_rank),
                'fs_rank': float(fs_rank),
                'hg_rank': float(hg_rank),
                'disentangle': float(disentangle),
                'effective_rank': float(er)
            })

    df = pd.DataFrame(all_results)

    # --- Plot 1: Disentanglement Score (Noise vs Alpha) ---
    plt.figure(figsize=(10, 7))
    for n_lvl in noise_levels:
        sub = df[df['noise'] == n_lvl]
        plt.plot(sub['alpha'], sub['disentangle'], marker='o', label=f'Noise {int(n_lvl*100)}%', linewidth=2)
    plt.title("Disentanglement: How well we separate Real vs Fake Stars", fontsize=14)
    plt.xlabel("Alpha", fontsize=12)
    plt.ylabel("Rank Gap (Fake - Real) | Higher is Better", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{base_dir}/01_disentanglement_power.png", dpi=200)
    plt.close()

    # --- Plot 2: Hidden Gem Recovery ---
    plt.figure(figsize=(10, 7))
    for n_lvl in noise_levels:
        sub = df[df['noise'] == n_lvl]
        plt.plot(sub['alpha'], sub['hg_rank'], marker='s', label=f'Noise {int(n_lvl*100)}%', linewidth=2)
    plt.gca().invert_yaxis()
    plt.title("Hidden Gem Recovery under Noise", fontsize=14)
    plt.xlabel("Alpha", fontsize=12)
    plt.ylabel("Avg Rank | Higher position is better", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{base_dir}/02_hidden_gem_recovery.png", dpi=200)
    plt.close()

    # --- Plot 3: Information Entropy (Effective Rank) ---
    plt.figure(figsize=(10, 7))
    for n_lvl in noise_levels:
        sub = df[df['noise'] == n_lvl]
        plt.plot(sub['alpha'], sub['effective_rank'], marker='^', label=f'Noise {int(n_lvl*100)}%', linewidth=2)
    plt.title("Data Structural Richness (Effective Rank)", fontsize=14)
    plt.xlabel("Alpha", fontsize=12)
    plt.ylabel("Effective Rank", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{base_dir}/03_structural_richness.png", dpi=200)
    plt.close()

    # Save JSON
    with open(f"{base_dir}/ace_properties_summary.json", 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\n✨ Heavy Noise Analysis complete! Plots saved in: {base_dir}")

if __name__ == "__main__":
    run_ace_analysis()
