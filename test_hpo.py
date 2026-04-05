import sys
import os
import json
import pandas as pd
import glob
sys.path.append(os.getcwd())
import __init__ as minirec

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
        'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '0.001 1000.0', 'log': True}]
    },
    {
        'name': 'IALS',
        'model_cfg': 'configs/models/ials.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '0.001 100.0', 'log': True},
                   {'name': 'alpha', 'type': 'float', 'range': '0.001 100.0', 'log': True}]
    },
    {
        'name' : 'puresvd',
        'model_cfg': 'configs/models/puresvd.yaml',
        'params': [{'name': 'k', 'type': 'int_for_k'}]
    },
    {'name' : 'gf_cf',
     'model_cfg': 'configs/models/gf_cf.yaml',
     'params': [{'name': 'alpha', 'type': 'float', 'range': '0.001 1.0', 'log': True},
                {'name': 'k', 'type': 'int_for_k'}]
     },
    {'name' : 'ips_lae',
     'model_cfg': 'configs/models/ips_lae.yaml',
     'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '0.001 1000.0', 'log': True},
                {'name': 'wbeta', 'type': 'float', 'range': '0.0 1.0', 'log': False}]
     },
    {
        'name' : 'dlae',
        'model_cfg': 'configs/models/dlae.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '0.001 1000.0', 'log': True},
                   {'name': 'dropout_p', 'type': 'float', 'range': '0.0 1.0', 'log': False}]
    },
    {
        'name' : 'ipsdlae',
        'model_cfg': 'configs/models/ipsdlae.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '0.001 1000.0', 'log': True},
                   {'name': 'dropout_p', 'type': 'float', 'range': '0.0 0.9', 'log': False}]
    },
    {
        'name' : 'ipswiener',
        'model_cfg': 'configs/models/ipswiener.yaml',
        'params': [{'name': 'reg_lambda', 'type': 'float', 'range': '0.001 1000.0', 'log': True}]
    }
]

hpo_cfg = {
    'metric': 'NDCG@20',
    'direction': 'max',
    'n_seeds': 3,
    'patience': 10
}

def generate_reports_from_files(output_root, hpo_results_root, main_metric):
    """
    디스크에 저장된 final_summary.json 파일들을 읽어 통합 리포트 생성
    """
    # 모든 final_summary.json 파일 탐색
    all_summary_files = glob.glob(os.path.join(hpo_results_root, "**", "final_summary.json"), recursive=True)
    
    collected_data = []
    main_metric_mean = f"{main_metric}_mean"
    
    for file_path in all_summary_files:
        # 경로에서 데이터셋명과 모델명 추출 (output/hpo_results/{dataset}/{model}/final_summary.json)
        # 운영체제별 경로 구분자 대응을 위해 정규화
        norm_path = os.path.normpath(file_path)
        parts = norm_path.split(os.sep)
        
        # 뒤에서부터 final_summary.json, model, dataset 순서
        if len(parts) < 3: continue
        d_name = parts[-3]
        m_name = parts[-2]
        
        try:
            with open(file_path, 'r') as f:
                summary = json.load(f)
            
            entry = {
                'dataset': d_name,
                'model': m_name
            }
            
            # 최적 파라미터 정보 추가 (첫 번째 시드 폴더 탐색)
            seed_dirs = glob.glob(os.path.join(os.path.dirname(file_path), "seed_*"))
            if seed_dirs:
                # 첫 번째 시드의 best_params.json 사용
                best_param_path = os.path.join(seed_dirs[0], "best_params.json")
                if os.path.exists(best_param_path):
                    with open(best_param_path, 'r') as f:
                        entry['best_params'] = json.dumps(json.load(f))
            
            # 메트릭 정보 업데이트
            entry.update(summary)
            collected_data.append(entry)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if not collected_data:
        print("No results found to report yet.")
        return

    df_all = pd.DataFrame(collected_data)
    
    # 정렬: 데이터셋별 가나다순, 그리고 메인 메트릭 성능 내림차순
    if main_metric_mean in df_all.columns:
        df_all = df_all.sort_values(by=['dataset', main_metric_mean], ascending=[True, False])
    
    # 1. 전체 통합 리포트 저장
    df_all.to_csv(os.path.join(output_root, 'all_hpo_results.csv'), index=False)
    
    # 2. 데이터셋별 개별 리포트 저장 및 출력
    print(f"\n{'='*30} HPO GLOBAL REPORT {'='*30}")
    for d_name in sorted(df_all['dataset'].unique()):
        df_d = df_all[df_all['dataset'] == d_name]
        df_d.to_csv(os.path.join(output_root, f'results_{d_name}.csv'), index=False)
        
        # 터미널 가독성을 위해 일부 컬럼만 출력
        display_cols = ['model', main_metric_mean, f"{main_metric}_std"]
        available_cols = [c for c in display_cols if c in df_d.columns]
        print(f"\n[Dataset: {d_name}]")
        print(df_d[available_cols].to_string(index=False))
    
    print(f"\n✨ Reports updated in '{output_root}' (Total {len(collected_data)} experiments found)")

# 3. 실험 실행 루프
hpo_results_root = 'output/hpo_results'
report_output_root = 'output/hpo_reports'
os.makedirs(report_output_root, exist_ok=True)

for d_cfg_path in datasets:
    d_name = os.path.basename(d_cfg_path).replace('.yaml', '')
    print(f"\n{'#'*80}\n### TARGET DATASET: {d_name} \n{'#'*80}")
    
    for exp in experiments:
        print(f"\n>>> Running HPO: {exp['name']} on {d_name}")
        
        hpo_cfg['params'] = exp['params']
        # HPO 실행 (내부에서 결과 파일 자동 저장)
        minirec.hporun(
            dataset_cfg=d_cfg_path,
            model_cfg=exp['model_cfg'],
            hpo_cfg=hpo_cfg,
            n_trials=20
        )
        
        # 모델 하나가 끝날 때마다 디스크의 파일들을 읽어 리포트 최신화
        generate_reports_from_files(report_output_root, hpo_results_root, hpo_cfg['metric'])

print(f"\n{'='*80}\n✨ All experiments finished! Final results are in '{report_output_root}'\n{'='*80}")
