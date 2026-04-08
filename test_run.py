import os
import sys
import json
import pandas as pd
import glob
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

# 3. 메인 실행 루프
print(f"\n{'='*80}\n### Starting Test Run (Default Configs) \n{'='*80}")

for d_cfg_path in datasets:
    d_name = os.path.basename(d_cfg_path).replace('.yaml', '')
    print(f"\n[Dataset: {d_name}]")
    for m_info in models:
        m_name, m_cfg = m_info['name'], m_info['cfg']
        print(f"  > Running: {m_name} ...", end=" ", flush=True)
        try:
            minirec.run(
                dataset_cfg=d_cfg_path,
                model_cfg=m_cfg,
                output_path=f"output/test_run/{d_name}/{m_name}"
            )
            print("Done.")
        except Exception as e:
            print(f"Error: {e}")
        
        # 모델 하나 끝날 때마다 리포트 갱신
        generate_global_report()

print(f"\n{'='*80}\n✨ All test runs finished!\n{'='*80}")
