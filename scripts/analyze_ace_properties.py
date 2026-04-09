import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from src.utils.config import merge_all_configs, load_yaml
from src.data.loader import DataLoader
from src.models.base import BaseModel

def compute_metrics(tensor):
    # Ensure CPU for stats
    t_np = tensor.detach().cpu().numpy()
    return {
        'std': float(np.std(t_np)),
        'skew': float(skew(t_np)),
        'kurtosis': float(kurtosis(t_np)),
        'min': float(np.min(t_np)),
        'max': float(np.max(t_np))
    }

def analyze_dataset(dataset_cfg_path):
    d_name = os.path.basename(dataset_cfg_path).replace('.yaml', '')
    print(f"\n>>> Analyzing ACE properties for dataset: {d_name}")
    
    # Load data
    cfg = load_yaml(dataset_cfg_path)
    dl = DataLoader(cfg)
    
    # Use BaseModel to get train matrix
    dummy_model = BaseModel(cfg, dl)
    X = dummy_model.get_train_matrix(dl).to(dummy_model.device)
    
    # G = X^T X
    G = torch.sparse.mm(X.t(), X.to_dense())
    
    eps = 1e-12
    d = G.diagonal()
    S = G.sum(dim=1)
    
    # 1. ACE Calculation
    log_ace = 2.0 * torch.log(d + eps) - torch.log(S + eps)
    ace = torch.exp(log_ace - log_ace.mean())
    
    # 2. Psi & Pure b_i (from AspirePure logic)
    scale = torch.pow(ace + eps, -0.5)
    G_tilde = G * scale.unsqueeze(1) * scale.unsqueeze(0)
    
    d_tilde = G_tilde.diagonal()
    S_tilde = G_tilde.sum(dim=1)
    log_psi = 2.0 * torch.log(d_tilde + eps) - torch.log(S_tilde + eps)
    psi = torch.exp(log_psi - log_psi.mean())
    
    b_pure = ace / (psi + eps)
    b_pure = torch.exp(torch.log(b_pure + eps) - torch.log(b_pure + eps).mean())

    # --- Experiment 1: ACE Distribution ---
    stats_ace = compute_metrics(ace)
    stats_bpure = compute_metrics(b_pure)
    
    # --- Experiment 3: Correlation (ACE vs S_i vs b_pure) ---
    s_norm = S / (S.mean() + eps)
    corr_ace_s = torch.corrcoef(torch.stack([ace.cpu(), s_norm.cpu()]))[0, 1].item()
    corr_ace_bpure = torch.corrcoef(torch.stack([ace.cpu(), b_pure.cpu()]))[0, 1].item()
    corr_bpure_s = torch.corrcoef(torch.stack([b_pure.cpu(), s_norm.cpu()]))[0, 1].item()

    # --- Experiment 4: Spectrum Analysis ---
    if G.shape[0] > 3000:
        indices = torch.randperm(G.shape[0])[:3000]
        G_sub = G[indices][:, indices]
        G_tilde_sub = G_tilde[indices][:, indices]
    else:
        G_sub = G
        G_tilde_sub = G_tilde
    
    print(f"  Computing eigenvalues on CPU...")
    evals_raw = torch.linalg.eigvalsh(G_sub.cpu())
    evals_ace = torch.linalg.eigvalsh(G_tilde_sub.cpu())
    
    evals_raw = evals_raw[evals_raw > 1e-7]
    evals_ace = evals_ace[evals_ace > 1e-7]
    
    evals_raw /= evals_raw.max()
    evals_ace /= evals_ace.max()
    
    return {
        'dataset': d_name,
        'ace_stats': stats_ace,
        'bpure_stats': stats_bpure,
        'corr_ace_s': corr_ace_s,
        'corr_ace_bpure': corr_ace_bpure,
        'corr_bpure_s': corr_bpure_s,
        'evals_raw': evals_raw.numpy(),
        'evals_ace': evals_ace.numpy()
    }

if __name__ == "__main__":
    datasets = [
        'configs/datasets/ml-100k.yaml',
        'configs/datasets/ml-1m.yaml',
        'configs/datasets/steam.yaml'
    ]
    
    results = []
    for d in datasets:
        try:
            results.append(analyze_dataset(d))
        except Exception as e:
            print(f"Failed to analyze {d}: {e}")
    
    if not results:
        sys.exit(1)

    print("\n" + "="*50)
    print("ANALYSIS REPORT")
    print("="*50)
    
    for res in results:
        print(f"\n[Dataset: {res['dataset']}]")
        print(f"ACE Stats: {res['ace_stats']}")
        print(f"Pure-b Stats: {res['bpure_stats']}")
        print(f"Correlations:")
        print(f"  corr(ACE, S_i): {res['corr_ace_s']:.4f}")
        print(f"  corr(ACE, b_pure): {res['corr_ace_bpure']:.4f}")
        print(f"  corr(b_pure, S_i): {res['corr_bpure_s']:.4f}")
        
    plt.figure(figsize=(15, 5))
    for i, res in enumerate(results):
        plt.subplot(1, 3, i+1)
        plt.plot(np.sort(res['evals_raw'])[::-1], label='Raw Gram')
        plt.plot(np.sort(res['evals_ace'])[::-1], label='ACE Normalized')
        plt.title(f"Spectrum: {res['dataset']}")
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend()
    
    os.makedirs('output/analysis', exist_ok=True)
    plt.tight_layout()
    plt.savefig('output/analysis/spectrum_comparison.png')
    print(f"\nPlot saved to output/analysis/spectrum_comparison.png")
