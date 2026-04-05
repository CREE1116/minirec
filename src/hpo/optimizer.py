import optuna
import copy
import os
import json

from src.utils.seed import set_seed

class BayesianOptimizer:
    def __init__(self, run_func, dataset_cfg, model_cfg, hpo_cfg):
        self.run_func = run_func
        self.dataset_cfg = dataset_cfg
        self.model_cfg = model_cfg
        self.hpo_cfg = hpo_cfg
        self.metric = hpo_cfg.get('metric', 'NDCG@10')
        self.maximize = hpo_cfg.get('direction', 'max') == 'max'
        self.n_seeds = hpo_cfg.get('n_seeds', 1)  # 멀티시드 지원 (기본 1)

    def objective(self, trial):
        model_cfg = copy.deepcopy(self.model_cfg)
        params = self.hpo_cfg.get('params', [])
        
        # Trial 기반 파라미터 제안
        for p in params:
            name = p['name']
            p_type = p['type']
            p_range = p['range']
            p_log = p.get('log', False)
            
            if p_type == 'float':
                low, high = map(float, p_range.split())
                val = trial.suggest_float(name, low, high, log=p_log)
            elif p_type == 'int':
                low, high = map(int, p_range.split())
                val = trial.suggest_int(name, low, high, log=p_log)
            elif p_type == 'categorical':
                choices = p_range if isinstance(p_range, list) else p_range.split()
                val = trial.suggest_categorical(name, choices)
            
            keys = name.split('.')
            d = model_cfg
            for k in keys[:-1]: d = d.setdefault(k, {})
            d[keys[-1]] = val
        
        # 멀티시드 성능 측정 및 통계 계산
        trial_seed_results = []
        base_seed = self.dataset_cfg.get('seed', 42)
        
        # Trial별 결과 저장을 위한 경로 생성
        trial_id = trial.number
        trial_output_path = os.path.join(self.model_cfg.get('output_path_override', 'output'), f'trial_{trial_id}')
        os.makedirs(trial_output_path, exist_ok=True)

        for i in range(self.n_seeds):
            current_seed = base_seed + i
            set_seed(current_seed)
            
            iter_dataset_cfg = copy.deepcopy(self.dataset_cfg)
            iter_dataset_cfg['seed'] = current_seed
            
            # 각 시드별 독립된 출력 경로 설정
            seed_output_path = os.path.join(trial_output_path, f'seed_{current_seed}')
            
            metrics = self.run_func(iter_dataset_cfg, model_cfg, output_path=seed_output_path, hpo_mode=True)
            trial_seed_results.append(metrics)
            
        # 모든 메트릭에 대해 평균과 표준편차 계산
        all_metric_names = trial_seed_results[0].keys()
        summary = {}
        for m in all_metric_names:
            vals = [res[m] for res in trial_seed_results if m in res]
            if vals:
                summary[f"{m}_mean"] = float(np.mean(vals))
                summary[f"{m}_std"] = float(np.std(vals))
        
        # Trial 요약 저장
        with open(os.path.join(trial_output_path, 'trial_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
            
        return summary.get(f"{self.metric}_mean", 0.0)

    def search(self, n_trials=20):
        study = optuna.create_study(direction='maximize' if self.maximize else 'minimize')
        study.optimize(self.objective, n_trials=n_trials)
        print(f"Best params: {study.best_params}")
        return study.best_params
