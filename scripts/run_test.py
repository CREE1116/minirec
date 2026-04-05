import os
import sys

# 프로젝트 루트(minirec의 상위 디렉토리)를 path에 추가하여 minirec 패키지를 찾을 수 있게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import minirec

# 설정 파일 경로 (minirec/configs 내부)
dataset_cfg = os.path.join(current_dir, "..", "configs", "dataset_ml100k.yaml")
output_path = os.path.join(project_root, "output", "minirec_restructured_test")

# 1. Closed-form Test (EASE)
print("\n" + "="*30 + "\nTesting Closed-form (EASE)\n" + "="*30)
ease_cfg = {
    'model': {'name': 'ease', 'reg_lambda': 500.0},
    'evaluation': {'metrics': ['NDCG', 'Recall'], 'top_k': [10], 'main_metric': 'NDCG', 'main_metric_k': 10}
}
metrics_ease = minirec.run(dataset_cfg, ease_cfg, output_path=os.path.join(output_path, 'ease'))
print(f"EASE Results: {metrics_ease}")

# 2. SGD Test (MF)
print("\n" + "="*30 + "\nTesting SGD (MF)\n" + "="*30)
mf_cfg = {
    'model': {'name': 'mf', 'embedding_dim': 64},
    'train': {'epochs': 3, 'batch_size': 1024, 'lr': 0.01, 'loss_type': 'pairwise', 'num_negatives': 1},
    'evaluation': {'metrics': ['NDCG', 'Recall'], 'top_k': [10], 'main_metric': 'NDCG', 'main_metric_k': 10}
}
metrics_mf = minirec.run(dataset_cfg, mf_cfg, output_path=os.path.join(output_path, 'mf'))
print(f"MF Results: {metrics_mf}")

# 3. HPO Test (MF LR Search)
print("\n" + "="*30 + "\nTesting HPO (MF LR Search)\n" + "="*30)
hpo_cfg = {
    'metric': 'NDCG@10',
    'direction': 'max',
    'params': [
        {'name': 'train.lr', 'type': 'float', 'range': '0.001 0.1', 'log': True}
    ]
}
best_params = minirec.hporun(dataset_cfg, mf_cfg, hpo_cfg, n_trials=3)
print(f"Best Params: {best_params}")
