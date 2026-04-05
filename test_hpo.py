import sys
import os
import json
import pandas as pd
sys.path.append(os.getcwd())
import __init__ as minirec

# 1. 대상 데이터셋 리스트
datasets = [
    'configs/datasets/ml-100k.yaml',
    # 'configs/datasets/steam.yaml',
    'configs/datasets/ml-1m.yaml',
]

# 2. 모델별 HPO 설정 리스트
experiments = [
    {
        'name': 'EASE',
        'model_cfg': 'configs/models/ease.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '0.001 1000.0', 'log': True}]
    },
    {
        'name': 'IALS',
        'model_cfg': 'configs/models/ials.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '0.001 100.0', 'log': True},
                   {'name': 'alpha', 'type': 'float', 'range': '0.001 100.0', 'log': True}]
    },
    {
        'name': 'ips_lae',
        'model_cfg': 'configs/models/ips_lae.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '0.001 1000.0', 'log': True},
                   {'name': 'wbeta', 'type': 'float', 'range': '0.001 1.0', 'log': True}]
    },
    {
        'name' : 'gf-cf',
        'model_cfg': 'configs/models/gf_cf.yaml',
        'params': [{'name': 'k', 'type': 'int_for_k', 'log': True},
                   {'name': 'alpha', 'type': 'float', 'range': '0.001 1.0', 'log': True}]
    },
    {
        'name' : 'puresvd',
        'model_cfg': 'configs/models/puresvd.yaml',
        'params': [{'name': 'k', 'type': 'int_for_k'}]
    },
    {
        'name' : 'dlae',
        'model_cfg': 'configs/models/dlae.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '0.001 1000.0', 'log': True},
                   {'name': 'dropout_p', 'type': 'float', 'range': '0.0 0.99', 'log': False}]
    },
    {
        'name' : 'lira',
        'model_cfg': 'configs/models/lira.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '0.001 1000.0', 'log': True}]
    }
]

hpo_cfg = {
    'metric': 'NDCG@20',
    'direction': 'max',
    'n_seeds': 3,
    'patience': 10
}

all_results = []
output_root = 'output/hpo_reports'
os.makedirs(output_root, exist_ok=True)

main_metric = hpo_cfg['metric']
main_metric_mean = f"{main_metric}_mean"

# 3. 실험 실행 (데이터셋 x 모델)
for d_cfg_path in datasets:
    d_name = os.path.basename(d_cfg_path).replace('.yaml', '')
    print(f"\n{'#'*80}\n### DATASET: {d_name} \n{'#'*80}")
    
    dataset_results = []
    
    for exp in experiments:
        print(f"\n>>> Model: {exp['name']} on {d_name}")
        
        hpo_cfg['params'] = exp['params']
        summary = minirec.hporun(
            dataset_cfg=d_cfg_path,
            model_cfg=exp['model_cfg'],
            hpo_cfg=hpo_cfg,
            n_trials=20
        )
        
        # 결과 수집
        result_entry = {
            'dataset': d_name,
            'model': exp['name']
        }
        # best_params_per_seed는 엑셀/CSV에 넣기 복잡하므로 문자열화하거나 별도 처리
        if 'best_params_per_seed' in summary:
            best_params = summary.pop('best_params_per_seed')
            result_entry['best_params'] = json.dumps(best_params[0]) # 첫 시드 기준 대표 파라미터만 기록
            
        result_entry.update(summary)
        dataset_results.append(result_entry)
        all_results.append(result_entry)

    # 데이터셋별 결과 정렬 및 저장
    df_dataset = pd.DataFrame(dataset_results)
    if main_metric_mean in df_dataset.columns:
        df_dataset = df_dataset.sort_values(by=main_metric_mean, ascending=(hpo_cfg['direction'] == 'min'))
    
    df_dataset.to_csv(os.path.join(output_root, f'results_{d_name}.csv'), index=False)
    print(f"\n[Summary: {d_name}]\n", df_dataset[['model', main_metric_mean, f"{main_metric}_std"]].to_string())

# 4. 통합 결과 저장
# 전체 통합 결과 정렬 (데이터셋별, 그리고 메트릭 순)
df_all = pd.DataFrame(all_results)
sort_cols = ['dataset']
if main_metric_mean in df_all.columns:
    sort_cols.append(main_metric_mean)

df_all = df_all.sort_values(by=sort_cols, ascending=[True, (hpo_cfg['direction'] == 'min')])

# JSON 저장
with open(os.path.join(output_root, 'all_hpo_results.json'), 'w') as f:
    json.dump(all_results, f, indent=4)

# CSV 저장
df_all.to_csv(os.path.join(output_root, 'all_hpo_results.csv'), index=False)

print(f"\n{'='*80}\n✨ All experiments finished! Results saved to '{output_root}'\n{'='*80}")
print(df_all.to_string())
