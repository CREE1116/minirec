import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_mnar_experiment():
    n_users, n_items = 2000, 1000
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. True Signal (v_i) 생성: 균일 분포
    base_prob = 0.1
    R_true = (torch.rand((n_users, n_items)) < base_prob).float()
    
    # 2. MNAR 오염 (아이템 편향 b_i + 유저 편향 q_u)
    strength = 1.0
    b_i = torch.pow(torch.linspace(0.1, 1.0, n_items), strength * 3)
    q_u = torch.pow(torch.rand(n_users), 2.0)
    
    P_obs = (b_i.unsqueeze(0) * q_u.unsqueeze(1))
    P_obs = P_obs / P_obs.mean() * 0.2
    P_obs = torch.clamp(P_obs, 0, 0.99)
    
    M = (torch.rand((n_users, n_items)) < P_obs).float()
    X_obs = R_true * M
    
    # 3. 지표 계산
    eps = 1e-12
    n_i = X_obs.sum(dim=0)
    n_u = X_obs.sum(dim=1)
    S_i = X_obs.t() @ n_u
    S_u = X_obs @ n_i
    
    SNR_i = (n_i**2) / (S_i + eps)
    SNR_u = (n_u**2) / (S_u + eps)
    
    alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
    results = []

    for alpha in alphas:
        # 방법 A: Bi-directional SNR 정규화 (D_u=SNR_u, D_i=SNR_i)
        W_u_snr = torch.pow(SNR_u + eps, -alpha/2.0)
        W_i_snr = torch.pow(SNR_i + eps, -alpha/2.0)
        X_snr = X_obs * W_u_snr.unsqueeze(1) * W_i_snr.unsqueeze(0)
        G_snr = X_snr.t() @ X_snr
        
        # 방법 B: Bi-directional Symmetric 정규화 (D_u=n_u, D_i=S_i)
        W_u_sym = torch.pow(n_u + eps, -alpha/2.0)
        W_i_sym = torch.pow(S_i + eps, -alpha/2.0)
        X_sym = X_obs * W_u_sym.unsqueeze(1) * W_i_sym.unsqueeze(0)
        G_sym = X_sym.t() @ X_sym
        
        # 평가
        for name, G_result in [('Bi-SNR', G_snr), ('Bi-Symmetric', G_sym)]:
            restored_energy = G_result.sum(dim=1)
            cv = restored_energy.std() / (restored_energy.mean() + eps)
            res_corr = np.corrcoef(restored_energy.numpy(), b_i.numpy())[0, 1]
            
            results.append({
                'Method': name,
                'Alpha': alpha,
                'CV': cv.item(),
                'Residual_Corr': res_corr
            })

    # 결과 분석
    df = pd.DataFrame(results)
    print("\n[Bi-directional SNR vs Bi-directional Symmetric]")
    pivot_cv = df.pivot(index='Alpha', columns='Method', values='CV')
    print("\n1. CV (Lower is better)")
    print(pivot_cv)
    
    pivot_corr = df.pivot(index='Alpha', columns='Method', values='Residual_Corr')
    print("\n2. Residual Bias Correlation (Target: 0.0)")
    print(pivot_corr)

    # 시각화
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for method in df['Method'].unique():
        sub = df[df['Method'] == method]
        plt.plot(sub['Alpha'], sub['CV'], marker='o', label=method)
    plt.title('Uniformity Recovery (CV)')
    plt.xlabel('Alpha')
    plt.ylabel('CV')
    plt.legend()

    plt.subplot(1, 2, 2)
    for method in df['Method'].unique():
        sub = df[df['Method'] == method]
        plt.plot(sub['Alpha'], sub['Residual_Corr'], marker='s', label=method)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residual Bias Correlation')
    plt.xlabel('Alpha')
    plt.ylabel('Correlation')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('mnar_symmetric_bidir_results.png')
    print("\nVisualization saved as 'mnar_symmetric_bidir_results.png'")

if __name__ == "__main__":
    run_mnar_experiment()
