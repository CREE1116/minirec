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

class ControlledSimulation:
    def __init__(self, n_users=1000, n_items=1500, latent_dim=20):
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim
        
        U = np.random.randn(n_users, latent_dim)
        V = np.random.randn(n_items, latent_dim)
        
        # --- Group Definition ---
        # High Quality: Boost norms of V
        V[0:100] *= 3.0 # Hidden Gems & Real Stars
        # Low Quality: Shrink norms of V
        V[100:150] *= 0.2 # Fake Stars
        
        self.true_scores = U @ V.T
        self.true_matrix = (self.true_scores > np.percentile(self.true_scores, 95)).astype(float)
        
        # 2. Exposure Bias (MNAR)
        exposure_prob = np.full(n_items, 0.05) 
        exposure_prob[50:150] = 0.8 # Real Stars & Fake Stars (High exposure)
        exposure_prob[0:50] = 0.01  # Hidden Gems (Low exposure)
        
        exposure_mask = np.random.rand(n_users, n_items) < exposure_prob[None, :]
        self.train_matrix = self.true_matrix * exposure_mask
        self.test_matrix = self.true_matrix * (self.train_matrix == 0)
        
        self.groups = {
            'HiddenGems': list(range(0, 50)),
            'RealStars': list(range(50, 100)),
            'FakeStars': list(range(100, 150)),
            'Background': list(range(150, 1500))
        }

    def get_loader(self):
        class MockLoader:
            def __init__(self, m, u, i):
                self.n_users, self.n_items = u, i
                r, c = m.nonzero()
                self.train_df = pd.DataFrame({'user_id': r, 'item_id': c})
                self.cache_filename = f"snr_sim_{np.random.randint(1e9)}"
        return MockLoader(self.train_matrix, self.n_users, self.n_items)

def run_snr_experiment_detailed():
    print("Setting up Enhanced SNR Preservation Experiment...")
    sim = ControlledSimulation()
    dl = sim.get_loader()
    
    base_dir = 'output/analytic/snr_bias_analysis'
    os.makedirs(base_dir, exist_ok=True)
    
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    group_stats = {g: {'ranks': [], 'recalls': [], 'rank_dist': []} for g in sim.groups.keys()}
    
    for a in tqdm(alphas, desc="Sweeping Alpha"):
        _GLOBAL_SPARSE_CACHE.clear()
        config = {'model': {'reg_lambda': 50.0, 'alpha': a, 'beta': 0.5}, 'device': 'cpu'}
        model = CausalAspire(config, dl)
        model.fit(dl)
        
        with torch.no_grad():
            scores = model.forward(torch.arange(sim.n_users)).numpy()
        scores[sim.train_matrix > 0] = -1e10
        
        all_ranks = np.argsort(np.argsort(-scores, axis=1), axis=1)
        
        for g_name, indices in sim.groups.items():
            g_ranks = all_ranks[:, indices]
            avg_rank = g_ranks.mean()
            group_stats[g_name]['ranks'].append(float(avg_rank))
            group_stats[g_name]['rank_dist'].append(g_ranks.flatten()) # Save distribution for specific alpha
            
            top_50 = np.argsort(-scores, axis=1)[:, :50]
            g_recalls = []
            for u in range(sim.n_users):
                gt = set(sim.test_matrix[u].nonzero()[0]) & set(indices)
                if not gt: continue
                hit = len(gt & set(top_50[u]))
                g_recalls.append(hit / len(gt))
            group_stats[g_name]['recalls'].append(float(np.mean(g_recalls)) if g_recalls else 0.0)

    # 1. Individual Plot: Rank Retention Trend
    plt.figure(figsize=(12, 8))
    for g_name, color, label in [('RealStars', 'tab:blue', 'Real Stars (Signal+Bias)'), 
                                 ('FakeStars', 'tab:red', 'Fake Stars (Noise Only)'),
                                 ('HiddenGems', 'tab:green', 'Hidden Gems (Signal Only)')]:
        plt.plot(alphas, group_stats[g_name]['ranks'], marker='o', label=label, color=color, linewidth=3)
    plt.gca().invert_yaxis()
    plt.title("Recommendation Rank Evolution: Signal Preservation vs Noise Suppression", fontsize=15)
    plt.xlabel("Normalization Strength (Alpha)", fontsize=12)
    plt.ylabel("Avg Rank (Lower is Better)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{base_dir}/01_rank_trends.png", dpi=200)
    plt.close()

    # 2. Individual Plot: Recall Performance
    plt.figure(figsize=(12, 8))
    for g_name, color in [('RealStars', 'tab:blue'), ('FakeStars', 'tab:red'), ('HiddenGems', 'tab:green')]:
        plt.plot(alphas, group_stats[g_name]['recalls'], marker='s', label=g_name, color=color, linewidth=3)
    plt.title("Recall@50 by Item Category across Alpha", fontsize=15)
    plt.xlabel("Alpha", fontsize=12)
    plt.ylabel("Recall@50", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{base_dir}/02_recall_trends.png", dpi=200)
    plt.close()

    # 3. Individual Plot: Rank Distribution (Box Plot for a=0.0 vs a=0.6 vs a=1.2)
    target_alpha_indices = [0, 3, 6] # 0.0, 0.6, 1.2
    for idx in target_alpha_indices:
        a_val = alphas[idx]
        plt.figure(figsize=(12, 8))
        data_to_plot = [group_stats[g]['rank_dist'][idx] for g in ['RealStars', 'FakeStars', 'HiddenGems']]
        plt.boxplot(data_to_plot, labels=['RealStars', 'FakeStars', 'HiddenGems'], patch_artist=True)
        plt.gca().invert_yaxis()
        plt.title(f"Rank Distribution per Group (Alpha={a_val})", fontsize=15)
        plt.ylabel("Rank (Lower is Better)")
        plt.savefig(f"{base_dir}/03_rank_dist_alpha_{a_val}.png", dpi=200)
        plt.close()

    # 4. Individual Plot: Signal-to-Noise Preservation Index
    # Ratio of RealStars Rank / FakeStars Rank
    plt.figure(figsize=(12, 8))
    snr_ratio = np.array(group_stats['FakeStars']['ranks']) / np.array(group_stats['RealStars']['ranks'])
    plt.plot(alphas, snr_ratio, marker='D', color='purple', linewidth=3)
    plt.title("Signal Preservation Index (FakeStars Rank / RealStars Rank)", fontsize=15)
    plt.ylabel("Higher = Better Noise Suppression")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{base_dir}/04_snr_preservation_index.png", dpi=200)
    plt.close()

    # Save metrics JSON (without full distributions to keep it small)
    json_summary = {
        'alphas': alphas,
        'metrics': {g: {'ranks': group_stats[g]['ranks'], 'recalls': group_stats[g]['recalls']} for g in group_stats}
    }
    with open(f"{base_dir}/snr_analysis_summary.json", 'w') as f:
        json.dump(json_summary, f, indent=4)
        
    print(f"\n✨ Detailed SNR Analysis complete! Individual plots saved in: {base_dir}")

if __name__ == "__main__":
    run_snr_experiment_detailed()
