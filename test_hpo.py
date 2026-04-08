import sys
import os
import json
import pandas as pd
import glob
sys.path.append(os.getcwd())
import __init__ as minirec
from src.utils.config import load_yaml

_eval_cfg = load_yaml('configs/evaluation.yaml')
_main_metric = f"{_eval_cfg.get('main_metric', 'NDCG')}@{_eval_cfg.get('main_metric_k', 20)}"

# 1. 대상 데이터셋 리스트
datasets = [
    'configs/datasets/ml-100k.yaml',
    'configs/datasets/ml-1m.yaml',
    'configs/datasets/steam.yaml',
]

# 2. 모델별 HPO 설정 리스트
experiments = [
    {
        'name': 'EASE',
        'model_cfg': 'configs/models/ease.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '0.01 100000.0', 'log': True}]
    },
    {
        'name' : 'ips_lae',
        'model_cfg': 'configs/models/ips_lae.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '0.1 10000.0', 'log': True},
                   {'name': 'propensity_gamma', 'type': 'float', 'range': '0.0 1.0'}]
    },
    {
        'name' : 'dlae',
        'model_cfg': 'configs/models/dlae.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '0.1 10000.0', 'log': True},
                   {'name': 'dropout_p', 'type': 'float', 'range': '0.0 0.9'}]
    },
    {
        'name' : 'aspire',
        'model_cfg': 'configs/models/aspire.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '0.1 10000.0', 'log': True},
                   {'name': 'alpha', 'type': 'float', 'range': '0.0 2.0'}]
    },
    {
        'name' : 'aspire_diag',
        'model_cfg': 'configs/models/aspire_diag.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '0.1 10000.0', 'log': True},
                   {'name': 'alpha', 'type': 'float', 'range': '0.0 2.0'}]
    },
    {
        'name' : 'aspire_rowsum',
        'model_cfg': 'configs/models/aspire_rowsum.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '0.1 10000.0', 'log': True},
                   {'name': 'alpha', 'type': 'float', 'range': '0.0 2.0'}]
    },
    {
        'name' : 'daspire',
        'model_cfg': 'configs/models/daspire.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '1.0 10000.0', 'log': True},
                   {'name': 'alpha', 'type': 'float', 'range': '0.0 2.0'},
                   {'name': 'dropout_p', 'type': 'float', 'range': '0.0 0.9'}]
    },
    {
        'name' : 'daspire_diag',
        'model_cfg': 'configs/models/daspire_diag.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '1.0 10000.0', 'log': True},
                   {'name': 'alpha', 'type': 'float', 'range': '0.0 2.0'},
                   {'name': 'dropout_p', 'type': 'float', 'range': '0.0 0.9'}]
    },
    {
        'name' : 'daspire_rowsum',
        'model_cfg': 'configs/models/daspire_rowsum.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '1.0 10000.0', 'log': True},
                   {'name': 'alpha', 'type': 'float', 'range': '0.0 2.0'},
                   {'name': 'dropout_p', 'type': 'float', 'range': '0.0 0.9'}]
    },
    {
        'name' : 'mf',
        'model_cfg': 'configs/models/mf.yaml',
        'params': [{'name': 'embedding_dim', 'type': 'int', 'range': '64 256'},
                   {'name': 'reg_weight', 'type': 'float', 'range': '1e-6 1e-1', 'log': True},
                   {'name': 'train.lr', 'type': 'float', 'range': '0.0001 0.01', 'log': True}]
    },
    {
        'name' : 'lightgcn',
        'model_cfg': 'configs/models/lightgcn.yaml',
        'params': [{'name': 'embedding_dim', 'type': 'int', 'range': '64 256'},
                   {'name': 'n_layers', 'type': 'int', 'range': '1 3'},
                   {'name': 'reg_weight', 'type': 'float', 'range': '1e-6 1e-1', 'log': True},
                   {'name': 'train.lr', 'type': 'float', 'range': '0.0001 0.01', 'log': True}]
    },
]

hpo_cfg = {
    'direction': 'max',
    'n_seeds': 5,
    'patience': 20
}

def generate_global_report():
    """
    hpo_results와 test_run 결과를 모두 취합하여 하나의 CSV로 저장
    """
    report_root = 'output/global_reports'
    os.makedirs(report_root, exist_ok=True)
    
    collected_data = []
    
    # 1. HPO 결과 수집
    hpo_files = glob.glob(os.path.join('output/hpo_results', "**", "final_summary.json"), recursive=True)
    for f in hpo_files:
        parts = os.path.normpath(f).split(os.sep)
        if len(parts) < 3: continue
        d_name, m_name = parts[-3], parts[-2]
        
        # 하이퍼파라미터 정보 로드 (best_params.json)
        hparams = {}
        # 시드별 폴더가 있는 경우 첫 번째 시드 정보를 대표로 사용
        best_param_files = glob.glob(os.path.join(os.path.dirname(f), "seed_*", "best_params.json"))
        if best_param_files:
            try:
                with open(best_param_files[0], 'r') as bp:
                    hparams = json.load(bp)
            except: pass

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
            
    # 2. Test Run 결과 수집
    test_run_files = glob.glob(os.path.join('output/test_run', "**", "metrics.json"), recursive=True)
    for f in test_run_files:
        parts = os.path.normpath(f).split(os.sep)
        if len(parts) < 3: continue
        d_name, m_name = parts[-3], parts[-2]
        with open(f, 'r') as j:
            data = json.load(j)
            entry = {
                'dataset': d_name, 
                'model': f"{m_name}(Default)", 
                'type': 'Default',
                'hyperparameters': 'Default YAML Config'
            }
            for k, v in data.items():
                entry[f"{k}_mean"] = v
                entry[f"{k}_std"] = 0.0
            collected_data.append(entry)
            
    if not collected_data:
        print("No results to aggregate.")
        return

    df = pd.DataFrame(collected_data)
    main_metric_mean = f"{_main_metric}_mean"
    
    if main_metric_mean in df.columns:
        df = df.sort_values(by=['dataset', main_metric_mean], ascending=[True, False])
    
    # 컬럼 순서 조정 (하이퍼파라미터를 앞쪽으로)
    cols = list(df.columns)
    if 'hyperparameters' in cols:
        cols.remove('hyperparameters')
        cols.insert(3, 'hyperparameters')
        df = df[cols]

    csv_path = os.path.join(report_root, 'integrated_report.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*30} GLOBAL INTEGRATED REPORT {'='*30}")
    for d_name in sorted(df['dataset'].unique()):
        df_d = df[df['dataset'] == d_name]
        display_cols = ['model', 'type', main_metric_mean]
        p_cols = [c for c in display_cols if c in df_d.columns]
        print(f"\n[Dataset: {d_name}]")
        print(df_d[p_cols].to_string(index=False))
    
    print(f"\n✨ Integrated report updated at '{csv_path}'")

# 3. 실험 실행 루프
hpo_results_root = 'output/hpo_results'
os.makedirs('output', exist_ok=True)

for d_cfg_path in datasets:
    d_name = os.path.basename(d_cfg_path).replace('.yaml', '')
    print(f"\n{'#'*80}\n### TARGET DATASET: {d_name} \n{'#'*80}")
    
    for exp in experiments:
        print(f"\n>>> Running HPO: {exp['name']} on {d_name}")
        hpo_cfg['params'] = exp['params']
        minirec.hporun(
            dataset_cfg=d_cfg_path,
            model_cfg=exp['model_cfg'],
            hpo_cfg=hpo_cfg,
            n_trials=50
        )
        # 각 모델이 끝날 때마다 전체 리포트 갱신
        generate_global_report()

print(f"\n{'='*80}\n✨ All HPO experiments finished!\n{'='*80}")
