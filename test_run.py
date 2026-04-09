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
    'configs/datasets/ml-100k.yaml',
    'configs/datasets/ml-1m.yaml',
    'configs/datasets/steam.yaml',
]

models = [
    # {'name': 'EASE',         'cfg': 'configs/models/ease.yaml'},
    # {'name': 'ips_lae',      'cfg': 'configs/models/ips_lae.yaml'},
    # {'name': 'dlae',         'cfg': 'configs/models/dlae.yaml'},
    # {'name': 'aspire',       'cfg': 'configs/models/aspire.yaml'},
    # {'name': 'aspire_diag',  'cfg': 'configs/models/aspire_diag.yaml'},
    # {'name': 'aspire_rowsum','cfg': 'configs/models/aspire_rowsum.yaml'},
    # {'name': 'daspire',      'cfg': 'configs/models/daspire.yaml'},
    # {'name': 'daspire_diag', 'cfg': 'configs/models/daspire_diag.yaml'},
    # {'name': 'daspire_rowsum','cfg': 'configs/models/daspire_rowsum.yaml'},
    {'name': 'mf',           'cfg': 'configs/models/mf.yaml'},
    {'name': 'lightgcn',     'cfg': 'configs/models/lightgcn.yaml'},
    {'name': 'gf_cf',        'cfg': 'configs/models/gf_cf.yaml'},
    {'name': 'bspm',         'cfg': 'configs/models/bspm.yaml'},
    {'name': 'turbocf',      'cfg': 'configs/models/turbocf.yaml'},
]

# 멀티시드 설정
seeds = [42, 43, 44, 45, 46]

def generate_global_report():
    """
    hpo_results와 test_run 결과를 모두 취합하여 하나의 CSV로 저장 (정렬 포함)
    """
    report_root = 'output/global_reports'
    os.makedirs(report_root, exist_ok=True)
    
    collected_data = []
    main_metric_mean = f"{_main_metric}_mean"
    
    # 1. HPO 결과 수집
    hpo_files = glob.glob(os.path.join('output/hpo_results', "**", "final_summary.json"), recursive=True)
    for f in hpo_files:
        parts = os.path.normpath(f).split(os.sep)
        if len(parts) < 3: continue
        d_name, m_name = parts[-3], parts[-2]
        
        hparams = {}
        best_param_files = glob.glob(os.path.join(os.path.dirname(f), "seed_*", "best_params.json"))
        if best_param_files:
            try:
                with open(best_param_files[0], 'r') as bp:
                    hparams = json.load(bp)
            except: pass

        try:
            with open(f, 'r') as j:
                data = json.load(j)
                entry = {
                    'dataset': d_name, 
                    'model': f"{m_name}(HPO)", 
                    'type': 'HPO',
                    'hyperparameters': json.dumps(hparams)
                }
                entry.update(data)
                collected_data.append(entry)
        except: pass
            
    # 2. Test Run 결과 수집 (멀티시드 대응)
    # output/test_run/{dataset}/{model}/seed_{seed}/metrics.json 구조를 탐색
    test_run_dirs = glob.glob(os.path.join('output/test_run', "*", "*"))
    for d in test_run_dirs:
        if not os.path.isdir(d): continue
        parts = os.path.normpath(d).split(os.sep)
        d_name, m_name = parts[-2], parts[-1]
        
        seed_files = glob.glob(os.path.join(d, "seed_*", "metrics.json"))
        if not seed_files:
            # 기존 단일 시드 결과 확인 (output/test_run/{dataset}/{model}/metrics.json)
            legacy_file = os.path.join(d, "metrics.json")
            if os.path.exists(legacy_file):
                seed_files = [legacy_file]
            else:
                continue

        results_per_key = {}
        for f in seed_files:
            try:
                with open(f, 'r') as j:
                    data = json.load(j)
                    for k, v in data.items():
                        results_per_key.setdefault(k, []).append(v)
            except: pass
        
        if not results_per_key: continue
        
        entry = {
            'dataset': d_name, 
            'model': f"{m_name}(Default)", 
            'type': 'Default',
            'hyperparameters': 'Default YAML Config'
        }
        for k, vals in results_per_key.items():
            entry[f"{k}_mean"] = np.mean(vals)
            entry[f"{k}_std"] = np.std(vals) if len(vals) > 1 else 0.0
        
        collected_data.append(entry)
            
    if not collected_data:
        print("No results to aggregate.")
        return

    df = pd.DataFrame(collected_data)
    
    # [정렬 로직] 데이터셋 오름차순, 메인 메트릭 내림차순
    if main_metric_mean in df.columns:
        df[main_metric_mean] = pd.to_numeric(df[main_metric_mean], errors='coerce')
        df = df.sort_values(by=['dataset', main_metric_mean], ascending=[True, False])
    
    # 컬럼 순서 조정
    cols = list(df.columns)
    if 'hyperparameters' in cols:
        cols.remove('hyperparameters')
        cols.insert(3, 'hyperparameters')
        df = df[cols]

    csv_path = os.path.join(report_root, 'integrated_report.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*30} GLOBAL INTEGRATED REPORT {'='*30}")
    for d_name in df['dataset'].unique():
        df_d = df[df['dataset'] == d_name]
        display_cols = ['model', 'type', main_metric_mean]
        p_cols = [c for c in display_cols if c in df_d.columns]
        print(f"\n[Dataset: {d_name}]")
        print(df_d[p_cols].to_string(index=False))
    
    print(f"\n✨ Integrated report updated at '{csv_path}'")

# 3. 메인 실행 루프
print(f"\n{'='*80}\n### Starting Multi-Seed Test Run (Default Configs) \n{'='*80}")

for d_cfg_path in datasets:
    d_name = os.path.basename(d_cfg_path).replace('.yaml', '')
    print(f"\n[Dataset: {d_name}]")
    for m_info in models:
        m_name, m_cfg = m_info['name'], m_info['cfg']
        print(f"  > Running: {m_name} ({len(seeds)} Seeds)")
        
        for seed in seeds:
            print(f"    Seed {seed} ...", end=" ", flush=True)
            try:
                # 데이터셋 설정 로드 후 시드 강제 주입
                d_cfg = load_yaml(d_cfg_path)
                d_cfg['seed'] = seed
                
                minirec.run(
                    dataset_cfg=d_cfg,
                    model_cfg=m_cfg,
                    output_path=f"output/test_run/{d_name}/{m_name}/seed_{seed}"
                )
                print("Done.")
            except Exception as e:
                print(f"Error: {e}")
        
        generate_global_report()

print(f"\n{'='*80}\n✨ All multi-seed test runs finished!\n{'='*80}")
