import optuna
import copy
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.seed import set_seed
from src.utils.config import merge_all_configs, load_yaml

class BayesianOptimizer:
    def __init__(self, run_func, dataset_cfg, model_cfg, hpo_cfg):
        self.run_func = run_func
        self.dataset_cfg = load_yaml(dataset_cfg) if isinstance(dataset_cfg, str) else dataset_cfg
        self.model_cfg = load_yaml(model_cfg) if isinstance(model_cfg, str) else model_cfg
        
        self.hpo_cfg = hpo_cfg
        self.metric = hpo_cfg.get('metric', 'NDCG@20')
        self.maximize = hpo_cfg.get('direction', 'max') == 'max'
        
        if 'seeds' in hpo_cfg:
            self.seeds = hpo_cfg['seeds']
        else:
            n_seeds = hpo_cfg.get('n_seeds', 3)
            self.seeds = [42 + i for i in range(n_seeds)]
            
        self.patience = hpo_cfg.get('patience', 20)
        self.params_list = hpo_cfg.get('params', [])

        # SGD 모델은 early stopping 기준(main_metric)과 HPO objective가 달라지면
        # _best_val_metrics에서 원하는 metric을 찾지 못해 objective가 0.0이 됨
        from src.utils.config import load_yaml as _load_yaml, merge_all_configs as _merge
        _eval_cfg = _merge(self.dataset_cfg, self.model_cfg).get('evaluation', {})
        _expected = f"{_eval_cfg.get('main_metric', 'NDCG')}@{_eval_cfg.get('main_metric_k', 20)}"
        if self.metric != _expected:
            print(f"[HPO WARNING] hpo_cfg metric='{self.metric}' differs from "
                  f"evaluation main_metric='{_expected}'. "
                  f"SGD models may silently return 0.0 as HPO objective. "
                  f"Consider aligning hpo_cfg metric with main_metric.")
        
        self.dataset_name = self.dataset_cfg.get('dataset_name', 'unknown_data').lower()
        m_name = self.model_cfg.get('model_name')
        if not m_name and 'model' in self.model_cfg:
            m_name = self.model_cfg['model'].get('name', self.model_cfg['model'].get('model_name', 'unknown_model'))
        self.model_name = m_name.lower()
        
        self.base_output_root = self.model_cfg.get('output_path_override', 'output')
        self.hpo_root = os.path.join(self.base_output_root, 'hpo_results', self.dataset_name, self.model_name)
        os.makedirs(self.hpo_root, exist_ok=True)
        
        # int_for_k 타입을 위한 max_k 계산용 DataLoader 초기화 (필요할 때만)
        self._max_k = None

    def get_max_k(self):
        if self._max_k is None:
            from src.data.loader import DataLoader
            # 임시 설정을 만들어 데이터 로더 로드
            temp_config = merge_all_configs(self.dataset_cfg, self.model_cfg)
            print(f"[HPO] Loading DataLoader to determine max_k...")
            temp_dl = DataLoader(temp_config)
            self._max_k = min(temp_dl.n_users, temp_dl.n_items) - 1
            print(f"[HPO] Determined max_k: {self._max_k}")
        return self._max_k

    def objective(self, trial, current_seed):
        # Deep copy the entire merged config if possible, or handle it carefully
        model_cfg = copy.deepcopy(self.model_cfg)
        current_params = {}
        
        for p_def in self.params_list:
            name = p_def['name']
            p_type = p_def.get('type', 'float')
            p_range = p_def.get('range')
            p_log = p_def.get('log', False)

            if p_type == 'float':
                if p_range is None: raise ValueError(f"Parameter '{name}' of type 'float' requires 'range' definition.")
                low, high = map(float, p_range.split())
                val = trial.suggest_float(name, low, high, log=p_log)
            elif p_type == 'int':
                if p_range is None: raise ValueError(f"Parameter '{name}' of type 'int' requires 'range' definition.")
                low, high = map(int, p_range.split())
                val = trial.suggest_int(name, low, high, log=p_log)
            elif p_type == 'int_for_k':
                max_k = self.get_max_k()
                val = trial.suggest_int(name, 1, max_k)
            elif p_type == 'categorical':
                if p_range is None: raise ValueError(f"Parameter '{name}' of type 'categorical' requires 'range' definition.")
                choices = p_range
                if isinstance(choices, str):
                    choices = choices.split()
                val = trial.suggest_categorical(name, choices)
            
            current_params[name] = val
            
            # 파라미터 경로 설정 최적화
            # 만약 이름에 '.'이 없다면 기본적으로 'model' 섹션에 속한 것으로 간주
            target_path = name if '.' in name else f"model.{name}"
            keys = target_path.split('.')
            
            d = model_cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = val
        
        iter_dataset_cfg = copy.deepcopy(self.dataset_cfg)
        iter_dataset_cfg['seed'] = current_seed
        set_seed(current_seed)
        
        # run_func internally merges dataset_cfg and model_cfg
        metrics = self.run_func(iter_dataset_cfg, model_cfg, hpo_mode=True)
        
        val = metrics.get(self.metric)
        if val is None:
            for k, v in metrics.items():
                if self.metric in k:
                    val = v
                    break
        
        return val if val is not None else 0.0

    def save_results(self, study, output_dir):
        """Save study results: CSV and visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            df = study.trials_dataframe()
            csv_path = os.path.join(output_dir, 'trials.csv')
            df.to_csv(csv_path, index=False)
        except Exception as e: print(f"Error saving CSV: {e}")

        try:
            import matplotlib.pyplot as plt
            trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            if len(trials) > 0:
                plt.figure(figsize=(10, 6))
                values = [t.value for t in trials]
                best_values = [np.max(values[:i+1]) if self.maximize else np.min(values[:i+1]) for i in range(len(values))]
                
                plt.plot(values, marker='o', alpha=0.5, label='Objective Value')
                plt.plot(best_values, color='red', linewidth=2, label='Best Value')
                plt.xlabel('Trial'); plt.ylabel('Metric Value')
                plt.title('Optimization History')
                plt.legend(); plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, 'optimization_history.png'))
                plt.close()

            if len(self.params_list) > 1:
                try:
                    importance = optuna.importance.get_param_importances(study)
                    plt.figure(figsize=(10, 6))
                    plt.barh(list(importance.keys()), list(importance.values()))
                    plt.xlabel('Importance'); plt.title('Hyperparameter Importance')
                    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'param_importance.png'))
                    plt.close()
                except Exception as e: print(f"Could not calculate importance: {e}")

            for p_def in self.params_list:
                p_name = p_def['name']
                try:
                    plt.figure(figsize=(8, 6))
                    x_vals = [t.params[p_name] for t in trials if p_name in t.params]
                    y_vals = [t.value for t in trials if p_name in t.params]
                    plt.scatter(x_vals, y_vals, alpha=0.6)
                    plt.xlabel(p_name); plt.ylabel('Metric Value')
                    plt.title(f'Slice Plot: {p_name}')
                    if p_def.get('log', False): plt.xscale('log')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(output_dir, f'slice_{p_name.replace(".", "_")}.png'))
                    plt.close()
                except Exception as e: print(f"Error plotting slice for {p_name}: {e}")
        except Exception as e: print(f"Error generating visualizations: {e}")

    def search(self, n_trials=20):
        all_seed_results = []
        all_best_params = []
        
        for seed in self.seeds:
            print(f"\n>>> Starting HPO for Seed: {seed}")
            seed_dir = os.path.join(self.hpo_root, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            
            sampler = optuna.samplers.TPESampler(seed=seed)
            study = optuna.create_study(direction='maximize' if self.maximize else 'minimize', sampler=sampler)
            
            class EarlyStoppingCallback:
                def __init__(self, patience, maximize):
                    self.patience, self.maximize = patience, maximize
                    self.best_score = -np.inf if maximize else np.inf
                    self.count = 0
                def __call__(self, study, trial):
                    if trial.state == optuna.trial.TrialState.COMPLETE:
                        is_better = trial.value > self.best_score if self.maximize else trial.value < self.best_score
                        if is_better: self.best_score, self.count = trial.value, 0
                        else:
                            self.count += 1
                            if self.count >= self.patience: study.stop()

            study.optimize(lambda t: self.objective(t, seed), n_trials=n_trials, 
                           callbacks=[EarlyStoppingCallback(self.patience, self.maximize)])
            
            self.save_results(study, seed_dir)
            
            print(f"Seed {seed} HPO finished. Evaluating BEST on Test set...")
            best_params = study.best_params
            all_best_params.append(best_params)
            
            # [추가] 각 시드별 최적 파라미터 별도 저장
            with open(os.path.join(seed_dir, 'best_params.json'), 'w') as f:
                json.dump(best_params, f, indent=4)

            best_model_cfg = copy.deepcopy(self.model_cfg)
            for name, val in best_params.items():
                target_path = name if '.' in name else f"model.{name}"
                keys = target_path.split('.')
                d = best_model_cfg
                for k in keys[:-1]: d = d.setdefault(k, {})
                d[keys[-1]] = val
            
            best_dataset_cfg = copy.deepcopy(self.dataset_cfg)
            best_dataset_cfg['seed'] = seed
            set_seed(seed)
            
            test_metrics = self.run_func(
                best_dataset_cfg, 
                best_model_cfg, 
                output_path=os.path.join(seed_dir, 'best_test_run'),
                hpo_mode=False 
            )
            all_seed_results.append(test_metrics)

        summary = self.report_final_results(all_seed_results)
        # 요약 결과에 시드별 최적 파라미터 정보 포함
        summary['best_params_per_seed'] = all_best_params
        return summary

    def report_final_results(self, results):
        if not results: return {}
        all_keys = sorted(list(results[0].keys()))
        summary = {}
        print(f"\n{'#'*60}\n FINAL HPO REPORT ({len(self.seeds)} Seeds) \n{'#'*60}")
        for key in all_keys:
            vals = [res[key] for res in results if key in res]
            mean, std = np.mean(vals), np.std(vals)
            summary[f"{key}_mean"], summary[f"{key}_std"] = float(mean), float(std)
            print(f"{key:<20}: {mean:.4f} ± {std:.4f}")
        summary_path = os.path.join(self.hpo_root, 'final_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"{'#'*60}\nSummary saved to: {summary_path}")
        return summary
