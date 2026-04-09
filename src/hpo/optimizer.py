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
        self.maximize = hpo_cfg.get('direction', 'max') == 'max'

        if 'seeds' in hpo_cfg:
            self.seeds = hpo_cfg['seeds']
        else:
            n_seeds = hpo_cfg.get('n_seeds', 3)
            self.seeds = [42 + i for i in range(n_seeds)]

        self.patience = hpo_cfg.get('patience', 20)
        self.params_list = hpo_cfg.get('params', [])
        self.mode = hpo_cfg.get('mode', 'bayesian').lower()

        # HPO objective는 항상 evaluation config의 main_metric을 사용
        _eval_cfg = merge_all_configs(self.dataset_cfg, self.model_cfg).get('evaluation', {})
        self.metric = f"{_eval_cfg.get('main_metric', 'NDCG')}@{_eval_cfg.get('main_metric_k', 20)}"
        
        self.dataset_name = self.dataset_cfg.get('dataset_name', 'unknown_data').lower()
        m_name = self.model_cfg.get('model_name')
        if not m_name and 'model' in self.model_cfg:
            m_name = self.model_cfg['model'].get('name', self.model_cfg['model'].get('model_name', 'unknown_model'))
        self.model_name = m_name.lower()
        
        self.base_output_root = self.model_cfg.get('output_path_override', 'output')
        self.hpo_root = os.path.join(self.base_output_root, 'hpo_results', self.dataset_name, self.model_name)
        os.makedirs(self.hpo_root, exist_ok=True)
        
        self._max_k = None

    def get_search_space(self):
        """Grid Search를 위한 search_space 생성"""
        search_space = {}
        for p_def in self.params_list:
            name = p_def['name']
            p_type = p_def.get('type', 'float')
            p_range = p_def.get('range')
            
            if p_type == 'categorical':
                choices = p_range
                if isinstance(choices, str):
                    choices = choices.split()
                search_space[name] = choices
            elif p_type == 'int':
                low, high = map(int, p_range.split())
                search_space[name] = list(range(low, high + 1))
            elif p_type == 'float':
                points = p_range.split()
                search_space[name] = [float(x) for x in points]
        return search_space

    def objective(self, trial, current_seed):
        model_cfg = copy.deepcopy(self.model_cfg)

        if self.mode == 'grid':
            # Grid mode: search_space에 정의된 값만 사용하도록 강제
            search_space = self.get_search_space()
            for name, choices in search_space.items():
                val = trial.suggest_categorical(name, choices)
                
                target_path = name if '.' in name else f"model.{name}"
                keys = target_path.split('.')
                d = model_cfg
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = val
        else:
            # Bayesian mode
            for p_def in self.params_list:
                name = p_def['name']
                p_type = p_def.get('type', 'float')
                p_range = p_def.get('range')
                p_log = p_def.get('log', False)

                if p_type == 'float':
                    low, high = map(float, p_range.split()[:2])
                    val = trial.suggest_float(name, low, high, log=p_log)
                elif p_type == 'int':
                    low, high = map(int, p_range.split()[:2])
                    val = trial.suggest_int(name, low, high, log=p_log)
                elif p_type == 'int_for_k':
                    max_k = self.get_max_k()
                    val = trial.suggest_int(name, 1, max_k)
                elif p_type == 'categorical':
                    choices = p_range
                    if isinstance(choices, str): choices = choices.split()
                    val = trial.suggest_categorical(name, choices)

                target_path = name if '.' in name else f"model.{name}"
                keys = target_path.split('.')
                d = model_cfg
                for k in keys[:-1]: d = d.setdefault(k, {})
                d[keys[-1]] = val
        
        iter_dataset_cfg = copy.deepcopy(self.dataset_cfg)
        iter_dataset_cfg['seed'] = current_seed
        set_seed(current_seed)
        
        metrics = self.run_func(iter_dataset_cfg, model_cfg, hpo_mode=True)
        val = metrics.get(self.metric)
        if val is None:
            for k, v in metrics.items():
                if self.metric in k:
                    val = v
                    break
        return val if val is not None else 0.0

    def get_max_k(self):
        if self._max_k is None:
            from src.data.loader import DataLoader
            temp_config = merge_all_configs(self.dataset_cfg, self.model_cfg)
            temp_dl = DataLoader(temp_config)
            data_full_rank = min(temp_dl.n_users, temp_dl.n_items) - 1
            self._max_k = min(data_full_rank, 2000)
        return self._max_k

    def save_results(self, study, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        try:
            df = study.trials_dataframe()
            df.to_csv(os.path.join(output_dir, 'trials.csv'), index=False)
        except: pass

    def search(self, n_trials=20):
        all_seed_results = []
        all_best_params = []
        
        for seed in self.seeds:
            print(f"\n>>> Starting HPO ({self.mode.upper()}) for Seed: {seed}")
            seed_dir = os.path.join(self.hpo_root, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            
            if self.mode == 'grid':
                search_space = self.get_search_space()
                sampler = optuna.samplers.GridSampler(search_space)
                actual_n_trials = None # GridSampler는 모든 조합 완료 시 자동 중단
                print(f"[Grid Mode] Search Space: {search_space}")
            else:
                sampler = optuna.samplers.TPESampler(seed=seed)
                actual_n_trials = n_trials

            study = optuna.create_study(direction='maximize' if self.maximize else 'minimize', sampler=sampler)
            
            # Early Stopping Callback (Grid 모드에서는 미사용)
            class EarlyStoppingCallback:
                def __init__(self, patience, maximize, mode):
                    self.patience, self.maximize, self.mode = patience, maximize, mode
                    self.best_score = -np.inf if maximize else np.inf
                    self.count = 0
                def __call__(self, study, trial):
                    if self.mode == 'grid': return
                    if trial.state == optuna.trial.TrialState.COMPLETE:
                        is_better = trial.value > self.best_score if self.maximize else trial.value < self.best_score
                        if is_better: self.best_score, self.count = trial.value, 0
                        else:
                            self.count += 1
                            if self.count >= self.patience: study.stop()

            study.optimize(lambda t: self.objective(t, seed), n_trials=actual_n_trials, 
                           callbacks=[EarlyStoppingCallback(self.patience, self.maximize, self.mode)])
            
            self.save_results(study, seed_dir)
            best_params = study.best_params
            all_best_params.append(best_params)
            
            with open(os.path.join(seed_dir, 'best_params.json'), 'w') as f:
                json.dump(best_params, f, indent=4)

            # Evaluate BEST
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
            
            test_metrics = self.run_func(best_dataset_cfg, best_model_cfg, 
                                        output_path=os.path.join(seed_dir, 'best_test_run'),
                                        hpo_mode=False)
            all_seed_results.append(test_metrics)

        summary = self.report_final_results(all_seed_results)
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
        return summary
