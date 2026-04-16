import sys
import os
import json
import pandas as pd
import numpy as np
import glob
sys.path.append(os.getcwd())
import __init__ as minirec
from src.utils.config import load_yaml

_eval_cfg = load_yaml('configs/evaluation.yaml')
_main_metric = f"{_eval_cfg.get('main_metric', 'NDCG')}@{_eval_cfg.get('main_metric_k', 20)}"

# 1. 대상 데이터셋 리스트 (폴더 이름)
datasets = [
    'ml-100k',
    'ml-1m',
    'steam',
]

# 2. 모델별 HPO 설정 리스트
experiments = [
    {
        'name': 'EASE',
        'model_cfg': 'configs/models/ease.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'min': 10, 'max': 10000, 'n_points': 10, 'scale': 'log'}]
    },
    {
        'name': 'fixed_aspire',
        'model_cfg': 'configs/models/fixed_aspire.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'min': 0.1, 'max': 1000, 'n_points': 10, 'scale': 'log'},
                   {'name': 'alpha', 'type': 'float', 'min': 0.0, 'max': 1.0, 'n_points': 11, 'scale': 'linear'}]
    },
    {
        'name' : 'ips_lae',
        'model_cfg': 'configs/models/ips_lae.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'min': 10, 'max': 10000, 'n_points': 10, 'scale': 'log'},
                   {'name': 'wbeta', 'type': 'float', 'min': 0.1, 'max': 0.9, 'n_points': 9, 'scale': 'linear'}]
    },
    {
        'name' : 'lae',
        'model_cfg': 'configs/models/lae.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'min': 10, 'max': 10000, 'n_points': 10, 'scale': 'log'}]
    },
    {
        'name' : 'rlae',
        'model_cfg': 'configs/models/RLAE.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'min': 10, 'max': 10000, 'n_points': 10, 'scale': 'log'},
                   {'name': 'b', 'type': 'float', 'min': 0.0, 'max': 1.0, 'n_points': 11, 'scale': 'linear'}]
    },
    {
        'name' : 'ease_dan',
        'model_cfg': 'configs/models/ease_dan.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'min': 0.1, 'max': 1000, 'n_points': 10, 'scale': 'log'},
                   {'name': 'alpha', 'type': 'float', 'min': 0.1, 'max': 1.0, 'n_points': 5, 'scale': 'linear'},
                   {'name': 'beta', 'type': 'float', 'min': 0.1, 'max': 1.0, 'n_points': 5, 'scale': 'linear'}]
    },
    {
        'name' : 'pmi_lae',
        'model_cfg': 'configs/models/pmi_lae.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'min': 0.1, 'max': 1000, 'n_points': 11, 'scale': 'log'},
                   {'name': 'alpha', 'type': 'float', 'min': 0.1, 'max': 2.0, 'n_points': 10, 'scale': 'linear'}]
    }
]

hpo_cfg = {
    'mode': 'grid',
    'direction': 'max',
    'n_seeds': 1,
}

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
                        # Extract hyperparameters from HPO summary
                        hparams = data.get('best_params_per_seed', [{}])[0]
                        entry = {
                            'dataset': d_name, 
                            'model': m_name, 
                            'type': 'HPO',
                            'hyperparameters': json.dumps(hparams)
                        }
                        # Remove parameters from data to avoid column duplication
                        clean_data = {k: v for k, v in data.items() if k != 'best_params_per_seed'}
                        entry.update(clean_data)
                        collected_data.append(entry)
                except: pass
            
    # 2. Test Run 결과 수집 (Val/Test 분리)
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
                    seed_dirs = glob.glob(os.path.join(m_path, "seed_*"))
                    for sd in seed_dirs:
                        m_file = os.path.join(sd, file_name)
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
    # Reorder columns to put main info first
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
        
        val_col = f"Best_VAL_{_main_metric}_mean"
        test_col = f"{_main_metric}_mean"
        
        cols = ['model', 'type']
        if val_col in df_d.columns: cols.append(val_col)
        if test_col in df_d.columns: cols.append(test_col)
        
        print(f"\n[Dataset: {d_name}]")
        print(df_d[cols].to_string(index=False))
    print(f"\n✨ Integrated report updated at '{csv_path}'")

if __name__ == "__main__":
    # 3. 실험 실행 루프
    hpo_results_root = 'output/hpo_results'
    os.makedirs('output', exist_ok=True)

    for d_name in datasets:
        print(f"\n{'#'*80}\n### TARGET DATASET: {d_name} \n{'#'*80}")
        
        for exp in experiments:
            print(f"\n>>> Running HPO: {exp['name']} on {d_name}")
            hpo_cfg['params'] = exp['params']
            minirec.hporun(
                dataset_name=d_name,
                model_cfg=exp['model_cfg'],
                hpo_cfg=hpo_cfg,
                n_trials=None # 그리드 모드에서는 모든 조합 시도
            )
            generate_global_report()

    print(f"\n{'='*80}\n✨ All HPO experiments finished!\n{'='*80}")
