import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from tqdm import tqdm

# Add root directory to path
sys.path.append(os.getcwd())

from src.models.ease import EASE
from src.models.causal_aspire import CausalAspire
from src.utils.sparse import _GLOBAL_SPARSE_CACHE, get_train_matrix_scipy

class SimulationDataset:
    """Generates synthetic data with MNAR bias."""
    def __init__(self, n_users=800, n_items=1200, latent_dim=20, exposure_gamma=1.5):
        self.n_users = n_users
        self.n_items = n_items
        
        # 1. True Preference (MCAR ground truth)
        U = np.random.randn(n_users, latent_dim)
        V = np.random.randn(n_items, latent_dim)
        scores = U @ V.T
        # Binarize to get True Preference Matrix
        self.true_pref = (scores > np.percentile(scores, 90)).astype(float)
        
        # 2. MNAR Exposure Mechanism
        # Items have power-law popularity
        item_base_pop = np.random.pareto(2.0, n_items) + 1.0
        item_base_pop /= item_base_pop.max()
        # Exposure probability depends on item popularity
        exposure_prob = np.power(item_base_pop, exposure_gamma)
        exposure_mask = np.random.rand(n_users, n_items) < exposure_prob
        
        # 3. Observed Interactions (Training set)
        self.train_matrix = self.true_pref * exposure_mask
        
        # 4. Test set (Pure preference from True Preference, items not in training)
        test_mask = (np.random.rand(n_users, n_items) < 0.2) * (self.train_matrix == 0)
        self.test_matrix = self.true_pref * test_mask

    def get_loader(self, matrix):
        class MockLoader:
            def __init__(self, m, u, i):
                self.n_users, self.n_items = u, i
                r, c = m.nonzero()
                self.train_df = pd.DataFrame({'user_id': r, 'item_id': c})
                self.cache_filename = f"mnar_sim_{np.random.randint(1e9)}"
        return MockLoader(matrix, self.n_users, self.n_items)

def calculate_gini(item_recs, n_items):
    counts = np.zeros(n_items)
    unique, cnts = np.unique(item_recs, return_counts=True)
    counts[unique] = cnts
    counts = np.sort(counts)
    n = n_items
    index = np.arange(1, n + 1)
    return ((2 * index - n - 1) * counts).sum() / (n * counts.sum() + 1e-12)

def evaluate_model(model, interaction_matrix, test_matrix, top_k=20):
    n_users = interaction_matrix.shape[0]
    n_items = interaction_matrix.shape[1]
    
    with torch.no_grad():
        u_idx = torch.arange(n_users)
        scores = model.forward(u_idx).numpy()
    
    # Mask training
    scores[interaction_matrix > 0] = -1e10
    
    top_items = np.argsort(scores, axis=1)[:, -top_k:]
    
    # Accuracy: Recall
    recalls = []
    for u in range(n_users):
        gt = set(test_matrix[u].nonzero()[0])
        if not gt: continue
        rec = set(top_items[u])
        recalls.append(len(gt & rec) / len(gt))
    
    # Fairness: Gini
    gini = calculate_gini(top_items.flatten(), n_items)
    
    # User Group Analysis (Experiment 3)
    user_activity = np.array(interaction_matrix.sum(axis=1)).flatten()
    quartiles = np.percentile(user_activity, [25, 50, 75])
    group_recalls = []
    for i in range(4):
        if i == 0: mask = user_activity <= quartiles[0]
        elif i == 1: mask = (user_activity > quartiles[0]) & (user_activity <= quartiles[1])
        elif i == 2: mask = (user_activity > quartiles[1]) & (user_activity <= quartiles[2])
        else: mask = user_activity > quartiles[2]
        
        group_idx = np.where(mask)[0]
        g_recalls = []
        for u in group_idx:
            gt = set(test_matrix[u].nonzero()[0])
            if not gt: continue
            rec = set(top_items[u])
            g_recalls.append(len(gt & rec) / len(gt))
        group_recalls.append(np.mean(g_recalls) if g_recalls else 0)

    return np.mean(recalls), gini, group_recalls

if __name__ == "__main__":
    print("Generating MNAR Simulation Data...")
    sim = SimulationDataset()
    train_dl = sim.get_loader(sim.train_matrix)
    
    # Experiment 1: Pareto Frontier (Alpha Sweep)
    alphas = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
    pareto_results = []
    
    print("\nSweeping Alpha for CausalAspire...")
    for a in alphas:
        _GLOBAL_SPARSE_CACHE.clear()
        config = {'model': {'reg_lambda': 200.0, 'alpha': a, 'beta': 0.5}, 'device': 'cpu'}
        model = CausalAspire(config, train_dl)
        model.fit(train_dl)
        acc, fair, groups = evaluate_model(model, sim.train_matrix, sim.test_matrix)
        pareto_results.append({'alpha': a, 'recall': acc, 'gini': fair, 'groups': groups})

    # Base EASE Comparison
    _GLOBAL_SPARSE_CACHE.clear()
    ease_config = {'model': {'reg_lambda': 200.0}, 'device': 'cpu'}
    ease_model = EASE(ease_config, train_dl)
    ease_model.fit(train_dl)
    e_acc, e_fair, e_groups = evaluate_model(ease_model, sim.train_matrix, sim.test_matrix)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Pareto Frontier
    recalls = [r['recall'] for r in pareto_results]
    ginis = [r['gini'] for r in pareto_results]
    axes[0].plot(ginis, recalls, marker='s', label='CausalAspire (Alpha Sweep)', color='blue')
    for i, a in enumerate(alphas):
        axes[0].annotate(f"a={a}", (ginis[i], recalls[i]))
    axes[0].scatter([e_fair], [e_acc], color='red', label='Standard EASE', s=100, zorder=5)
    axes[0].set_title('Experiment 1: Accuracy-Fairness Pareto Frontier')
    axes[0].set_xlabel('Gini Index (Lower is Better/Fairer)')
    axes[0].set_ylabel('Recall@20 (Higher is Better)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Plot 2: User Utility Inequality
    labels = ['Q1 (Inactive)', 'Q2', 'Q3', 'Q4 (Active)']
    x = np.arange(len(labels))
    width = 0.35
    
    # Use best alpha (say 1.2) for comparison
    best_idx = 3 # alpha 1.2
    axes[1].bar(x - width/2, e_groups, width, label='EASE', color='red', alpha=0.6)
    axes[1].bar(x + width/2, pareto_results[best_idx]['groups'], width, label=f'CausalAspire (a={alphas[best_idx]})', color='blue', alpha=0.6)
    axes[1].set_title('Experiment 3: User Utility (Recall per Activity Group)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('Recall@20')
    axes[1].legend()

    plt.tight_layout()
    output_path = 'output/mnar_analysis_report.png'
    os.makedirs('output', exist_ok=True)
    plt.savefig(output_path)
    print(f"\n✨ Analysis complete! Report saved at: {output_path}")
