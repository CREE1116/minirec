import os
import sys
import json
import pandas as pd
import glob
import numpy as np
sys.path.append(os.getcwd())
import __init__ as minirec
from src.utils.config import load_yaml

# 1. 설정 로드 (메트릭 확인용)
_eval_cfg = load_yaml('configs/evaluation.yaml')
_main_metric = f"{_eval_cfg.get('main_metric', 'NDCG')}@{_eval_cfg.get('main_metric_k', 20)}"

# 2. 대상 데이터셋 및 모델
datasets = [
    'ml-100k',
    'ml-1m',
    'steam',
]

models = [
    {'name': 'mf',           'cfg': 'configs/models/mf.yaml'},
    {'name': 'lightgcn',     'cfg': 'configs/models/lightgcn.yaml'},
    {'name': 'gf_cf',        'cfg': 'configs/models/gf_cf.yaml'},
    {'name': 'bspm',         'cfg': 'configs/models/bspm.yaml'},
    {'name': 'turbocf',      'cfg': 'configs/models/turbocf.yaml'},
    {'name': 'mf_ips',       'cfg': 'configs/models/mf_ips.yaml'},
    {'name': 'dr_jl',        'cfg': 'configs/models/dr_jl.yaml'},
    {'name': 'pmi_lae',      'cfg': 'configs/models/pmi_lae.yaml'},
]

# 5-Seed 설정
seeds = [42, 43, 44, 45, 46]

def generate_global_report():
    """
    hpo_results와 test_run 결과를 모두 취합하여 하나의 CSV로 저장 (Val/Test 대응)
    """
    report_root = 'output/global_reports'
    os.makedirs(report_root, exist_ok=True)
    
    collected_data = []
    
    # 1. HPO 결과 수집
    hpo_results_base = 'output/hpo_results'
    if os.path.exists(hpo_results_base):
        for d_name in os.listdir(hpo_results_base):
            d_path = os.path.join(hpo_results_base, d_name)
            if not os.path.isdir(d_path): continue
            for m_name in os.listdir(d_path):
                m_path = os.path.join(d_path, m_name)
                summary_file = os.path.join(m_path, "final_summary.json")
                if not os.path.exists(summary_file): continue
                
                try:
                    with open(summary_file, 'r') as j:
                        data = json.load(j)
                        hparams = data.get('best_params_per_seed', [{}])[0]
                        entry = {
                            'dataset': d_name, 
                            'model': m_name, 
                            'type': 'HPO',
                            'hyperparameters': json.dumps(hparams)
                        }
                        clean_data = {k: v for k, v in data.items() if k != 'best_params_per_seed'}
                        entry.update(clean_data)
                        collected_data.append(entry)
                except: pass
            
    # 2. Test Run 결과 수집 (5-시드 취합, Val/Test 분리)
    test_run_base = 'output/test_run'
    if os.path.exists(test_run_base):
        for d_name in os.listdir(test_run_base):
            d_path = os.path.join(test_run_base, d_name)
            if not os.path.isdir(d_path): continue
            for m_name in os.listdir(d_path):
                m_path = os.path.join(d_path, m_name)
                if not os.path.isdir(m_path): continue

                for m_type in ['VAL', 'TEST']:
                    file_name = "val_metrics.json" if m_type == 'VAL' else "metrics.json"
                    seed_metrics = []
                    for s in seeds:
                        m_file = os.path.join(m_path, f"seed_{s}", file_name)
                        if os.path.exists(m_file):
                            try:
                                with open(m_file, 'r') as f: seed_metrics.append(json.load(f))
                            except: pass
                    
                    if not seed_metrics: continue

                    entry = {
                        'dataset': d_name, 
                        'model': m_name, 
                        'type': f'Default_{m_type}',
                        'hyperparameters': 'Default'
                    }
                    
                    all_keys = seed_metrics[0].keys()
                    for k in all_keys:
                        vals = [m[k] for m in seed_metrics if k in m]
                        entry[f"{k}_mean"] = np.mean(vals)
                        entry[f"{k}_std"] = np.std(vals) if len(vals) > 1 else 0.0
                    
                    collected_data.append(entry)
            
    if not collected_data: return

    df = pd.DataFrame(collected_data)
    id_cols = ['dataset', 'model', 'type', 'hyperparameters']
    other_cols = sorted([c for c in df.columns if c not in id_cols])
    df = df[id_cols + other_cols]

    main_metric_mean = f"{_main_metric}_mean"
    if main_metric_mean in df.columns:
        df = df.sort_values(by=['dataset', main_metric_mean], ascending=[True, False])

    csv_path = os.path.join(report_root, 'integrated_report.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*30} GLOBAL REPORT {'='*30}")
    for d_name in df['dataset'].unique():
        df_d = df[df['dataset'] == d_name]
        p_cols = ['model', 'type', main_metric_mean]
        p_cols = [c for c in p_cols if c in df_d.columns]
        print(f"\n[Dataset: {d_name}]")
        print(df_d[p_cols].to_string(index=False))
    print(f"\n✨ Integrated report updated at '{csv_path}'")

if __name__ == "__main__":
    print(f"\n{'='*80}\n### Starting 5-Seed Multi-Test Run \n{'='*80}")

    for d_name in datasets:
        print(f"\n{'#'*80}\n### TARGET DATASET: {d_name} \n{'#'*80}")
        
        for m_info in models:
            m_name, m_cfg = m_info['name'], m_info['cfg']
            print(f"\n>> Model: {m_name}")
            
            for seed in seeds:
                print(f"  Seed {seed} ...", end=" ", flush=True)
                try:
                    # evaluation 설정 로드 및 시드 오버라이드
                    current_eval_cfg = load_yaml('configs/evaluation.yaml')
                    current_eval_cfg['seed'] = seed
                    
                    # run 함수 호출
                    minirec.run(
                        dataset_name=d_name,
                        model_cfg=m_cfg,
                        output_path=f"output/test_run/{d_name}/{m_name}/seed_{seed}"
                    )
                    print("Done.")
                except Exception as e:
                    print(f"Error: {e}")
            
            generate_global_report()

    print(f"\n{'='*80}\n✨ All test runs finished!\n{'='*80}")
