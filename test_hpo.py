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

# 1. 대상 데이터셋 리스트
datasets = [
    'configs/datasets/ml-100k.yaml',
    'configs/datasets/ml-1m.yaml',
    'configs/datasets/steam.yaml',
    # 'configs/datasets/ml-20m.yaml',
    # 'configs/datasets/yelp2018.yaml',
    # 'configs/datasets/gowalla.yaml'
    # 'configs/datasets/amazon_electronics.yaml',
    # 'configs/datasets/amazon_books.yaml'
]

# 2. 모델별 HPO 설정 리스트
experiments = [
    {
        'name': 'EASE',
        'model_cfg': 'configs/models/ease.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'min': 100, 'max': 10000, 'n_points': 10, 'scale': 'log'}]
    },
    {
        'name': 'causal_aspire',
        'model_cfg': 'configs/models/causal_aspire.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'min': 1, 'max': 100, 'n_points': 10, 'scale': 'log'},
                    # {'name': 'alpha', 'type': 'float', 'min': 0.1, 'max': 1.0, 'n_points': 10, 'scale': 'linear'}
                   ]
    },
        {
        'name': 'causal_aspire_dropout',
        'model_cfg': 'configs/models/causal_aspire_dropout.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'min': 1, 'max': 100, 'n_points': 10, 'scale': 'log'},
                    #  {'name': 'alpha', 'type': 'float', 'min': 0.1, 'max': 1.0, 'n_points': 10, 'scale': 'linear'},
                    {'name': 'dropout_p', 'type': 'float', 'min': 0.1, 'max': 0.9, 'n_points': 9, 'scale': 'linear'}]
    },
    {
        'name' : 'ips_lae',
        'model_cfg': 'configs/models/ips_lae.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'min': 100, 'max': 10000, 'n_points': 10, 'scale': 'log'},
                   {'name': 'wbeta', 'type': 'float', 'min': 0.1, 'max': 0.9, 'n_points': 9, 'scale': 'linear'}]
    },
    # {
    #     'name' : 'dlae',
    #     'model_cfg': 'configs/models/dlae.yaml',
    #     'params': [{'name': 'reg_lambda', 'type': 'float', 'min': 100, 'max': 10000, 'n_points': 10, 'scale': 'log'},
    #                {'name': 'dropout_p', 'type': 'float', 'min': 0.1, 'max': 0.9, 'n_points': 9, 'scale': 'linear'}]
    # },
    {
        'name' : 'lae',
        'model_cfg': 'configs/models/lae.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'min': 100, 'max': 10000, 'n_points': 10, 'scale': 'log'}]
    },
    {
        'name' : 'aspire_ips',
        'model_cfg': 'configs/models/aspire_ips.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'min': 10, 'max': 1000, 'n_points': 10, 'scale': 'log'},
                   {'name': 'alpha', 'type': 'float', 'min': 0.2, 'max': 2.0, 'n_points': 10, 'scale': 'linear'}]
    },
    {
        'name' : 'rlae',
        'model_cfg': 'configs/models/RLAE.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'min': 100, 'max': 10000, 'n_points': 10, 'scale': 'log'},
                   {'name': 'b', 'type': 'float', 'min': 0.0, 'max': 1.0, 'n_points': 11, 'scale': 'linear'}]
    },
    # {
    #     'name' : 'rdlae',
    #     'model_cfg': 'configs/models/rdlae.yaml',
    #     'params': [{'name': 'reg_lambda', 'type': 'float', 'min': 10, 'max': 1000, 'n_points': 10, 'scale': 'log'},
    #                {'name': 'dropout_p', 'type': 'float', 'min': 0.1, 'max': 0.9, 'n_points': 9, 'scale': 'linear'}]
    # },
    {
        'name' : 'ease_dan',
        'model_cfg': 'configs/models/ease_dan.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'min': 0.1, 'max': 1000, 'n_points': 11, 'scale': 'log'},
                   {'name': 'alpha', 'type': 'float', 'min': 0.1, 'max': 1.0, 'n_points': 5, 'scale': 'linear'},
                   {'name': 'beta', 'type': 'float', 'min': 0.1, 'max': 1.0, 'n_points': 5, 'scale': 'linear'}]
    }
]

hpo_cfg = {
    'mode': 'grid',
    'direction': 'max',
    'n_seeds': 1,
}

def generate_global_report():
    """
    hpo_results와 test_run 결과를 모두 취합하여 하나의 CSV로 저장
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
                
                hparams = {}
                seed_dirs = sorted(glob.glob(os.path.join(m_path, "seed_*")))
                if seed_dirs:
                    bp_path = os.path.join(seed_dirs[0], "best_params.json")
                    if os.path.exists(bp_path):
                        try:
                            with open(bp_path, 'r') as bp: hparams = json.load(bp)
                        except: pass

                try:
                    with open(summary_file, 'r') as j:
                        data = json.load(j)
                        data.pop('best_params_per_seed', None)
                        entry = {
                            'dataset': d_name, 
                            'model': m_name, 
                            'type': 'HPO',
                            'hyperparameters': json.dumps(hparams)
                        }
                        entry.update(data)
                        collected_data.append(entry)
                except: pass
            
    # 2. Test Run 결과 수집
    test_run_base = 'output/test_run'
    if os.path.exists(test_run_base):
        for d_name in os.listdir(test_run_base):
            d_path = os.path.join(test_run_base, d_name)
            if not os.path.isdir(d_path): continue
            for m_name in os.listdir(d_path):
                m_path = os.path.join(d_path, m_name)
                if not os.path.isdir(m_path): continue

                seed_files = glob.glob(os.path.join(m_path, "seed_*", "metrics.json"))
                if not seed_files: continue

                metrics_agg = {}
                for sf in seed_files:
                    try:
                        with open(sf, 'r') as j:
                            m_data = json.load(j)
                            for k, v in m_data.items():
                                metrics_agg.setdefault(k, []).append(v)
                    except: pass
                
                if not metrics_agg: continue
                entry = {
                    'dataset': d_name, 
                    'model': m_name, 
                    'type': 'Default',
                    'hyperparameters': 'Default'
                }
                for k, vals in metrics_agg.items():
                    entry[f"{k}_mean"] = np.mean(vals)
                    entry[f"{k}_std"] = np.std(vals) if len(vals) > 1 else 0.0
                collected_data.append(entry)
            
    if not collected_data:
        return

    df = pd.DataFrame(collected_data)
    
    # ID 컬럼 배치 및 메트릭 정렬
    id_cols = ['dataset', 'model', 'type', 'hyperparameters']
    other_cols = sorted([c for c in df.columns if c not in id_cols])
    df = df[id_cols + other_cols]

    main_metric_mean = f"{_main_metric}_mean"
    if main_metric_mean in df.columns:
        df[main_metric_mean] = pd.to_numeric(df[main_metric_mean], errors='coerce')
        df = df.sort_values(by=['dataset', main_metric_mean], ascending=[True, False])

    csv_path = os.path.join(report_root, 'integrated_report.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*30} GLOBAL REPORT {'='*30}")
    for d_name in df['dataset'].unique():
        df_d = df[df['dataset'] == d_name]
        
        # 표시할 컬럼들 (기본 메트릭 + 언바이어스드 메트릭)
        k = _eval_cfg.get('main_metric_k', 20)
        display_cols = ['model', 'type', f'NDCG@{k}_mean', f'uNDCG@{k}_mean', f'Recall@{k}_mean', f'uRecall@{k}_mean']
        
        # 데이터프레임에 실제 존재하는 컬럼만 필터링
        p_cols = [c for c in display_cols if c in df_d.columns]
        if 'model' not in p_cols: p_cols.insert(0, 'model')
        
        print(f"\n[Dataset: {d_name}]")
        print(df_d[p_cols].to_string(index=False))
    print(f"\n✨ Integrated report updated at '{csv_path}'")

if __name__ == "__main__":
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
                n_trials=None # 그리드 모드에서는 모든 조합 시도
            )
            generate_global_report()

    print(f"\n{'='*80}\n✨ All HPO experiments finished!\n{'='*80}")
