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
    def __init__(self, run_func, dataset_name, model_cfg, hpo_cfg):
        self.run_func = run_func
        self.dataset_name = dataset_name
        self.model_cfg = load_yaml(model_cfg) if isinstance(model_cfg, str) else model_cfg
        
        self.hpo_cfg = hpo_cfg
        self.maximize = hpo_cfg.get('direction', 'max') == 'max'
        self.use_test_for_hpo = hpo_cfg.get('use_test_for_hpo', False)

        # HPO trials seeds
        if 'seeds' in hpo_cfg:
            self.seeds = hpo_cfg['seeds']
        else:
            n_seeds = hpo_cfg.get('n_seeds', 1) 
            self.seeds = [42 + i for i in range(n_seeds)]

        self.patience = hpo_cfg.get('patience', 20)
        self.params_list = hpo_cfg.get('params', [])
        self.mode = hpo_cfg.get('mode', 'bayesian').lower()

        # Load evaluation config for metric and seeds
        eval_cfg = load_yaml('configs/evaluation.yaml')
        self.metric_name = eval_cfg.get('main_metric', 'NDCG')
        self.metric_k = eval_cfg.get('main_metric_k', 20)
        self.metric = f"{self.metric_name}@{self.metric_k}"
        
        m_name = self.model_cfg.get('model_name')
        if not m_name and 'model' in self.model_cfg:
            m_name = self.model_cfg['model'].get('name', self.model_cfg['model'].get('model_name', 'unknown_model'))
        self.model_name = m_name.lower()
        
        self.base_output_root = self.model_cfg.get('output_path_override', 'output')
        self.hpo_root = os.path.join(self.base_output_root, 'hpo_results', self.dataset_name, self.model_name)
        os.makedirs(self.hpo_root, exist_ok=True)
        
        self._max_k = None

    def get_search_space(self):
        search_space = {}
        for p_def in self.params_list:
            name = p_def['name']
            p_type = p_def.get('type', 'float')
            
            if 'min' in p_def and 'max' in p_def and 'n_points' in p_def:
                low, high, n = float(p_def['min']), float(p_def['max']), int(p_def['n_points'])
                scale = p_def.get('scale', 'linear').lower()
                if scale == 'log': values = np.logspace(np.log10(low), np.log10(high), num=n).tolist()
                else: values = np.linspace(low, high, num=n).tolist()
                if p_type == 'int': values = sorted(list(set([int(round(v)) for v in values])))
                search_space[name] = values
                continue

            p_range = p_def.get('range')
            if p_type == 'categorical':
                choices = p_range
                if isinstance(choices, str): choices = choices.split()
                search_space[name] = choices
            elif p_type == 'int':
                parts = p_range.split()
                if len(parts) == 2:
                    low, high = map(int, parts)
                    search_space[name] = list(range(low, high + 1))
                else: search_space[name] = [int(x) for x in parts]
            elif p_type == 'float':
                points = p_range.split()
                search_space[name] = [float(x) for x in points]
        return search_space

    def objective(self, trial, current_seed, data_loader=None):
        model_cfg = copy.deepcopy(self.model_cfg)

        if self.mode == 'grid':
            search_space = self.get_search_space()
            for name, choices in search_space.items():
                val = trial.suggest_categorical(name, choices)
                keys = (name if '.' in name else f"model.{name}").split('.')
                d = model_cfg
                for k in keys[:-1]: d = d.setdefault(k, {})
                d[keys[-1]] = val
        else:
            for p_def in self.params_list:
                name, p_type, p_range, p_log = p_def['name'], p_def.get('type', 'float'), p_def.get('range'), p_def.get('log', False)
                if p_type == 'float':
                    low, high = map(float, p_range.split()[:2])
                    val = trial.suggest_float(name, low, high, log=p_log)
                elif p_type == 'int':
                    low, high = map(int, p_range.split()[:2])
                    val = trial.suggest_int(name, low, high, log=p_log)
                elif p_type == 'int_for_k': val = trial.suggest_int(name, 1, self.get_max_k(data_loader))
                elif p_type == 'categorical':
                    choices = p_range
                    if isinstance(choices, str): choices = choices.split()
                    val = trial.suggest_categorical(name, choices)
                keys = (name if '.' in name else f"model.{name}").split('.')
                d = model_cfg
                for k in keys[:-1]: d = d.setdefault(k, {})
                d[keys[-1]] = val
        
        metrics = self.run_func(self.dataset_name, model_cfg, hpo_mode=True, use_test_for_hpo=self.use_test_for_hpo, data_loader=data_loader)
        val = metrics.get(self.metric)
        if val is None:
            for k, v in metrics.items():
                if self.metric in k: val = v; break
        return val if val is not None else 0.0

    def get_max_k(self, data_loader=None):
        if self._max_k is None:
            if data_loader is not None:
                self._max_k = min(min(data_loader.n_users, data_loader.n_items) - 1, 2000)
            else:
                from src.data.loader import DataLoader
                temp_dl = DataLoader({'dataset_name': self.dataset_name})
                self._max_k = min(min(temp_dl.n_users, temp_dl.n_items) - 1, 2000)
        return self._max_k

    def save_results(self, study, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        try:
            df = study.trials_dataframe()
            df.to_csv(os.path.join(output_dir, 'trials.csv'), index=False)
        except: pass

    def search(self, n_trials=20):
        from src.utils.sparse import clear_sparse_cache
        from src.data.loader import DataLoader
        
        clear_sparse_cache() # New dataset/model start
        
        all_seed_results = []
        all_best_val_scores = []
        all_best_params = []
        self._max_k = None
        
        # Instantiate DataLoader once for this dataset
        print(f"[HPO] Pre-loading DataLoader for {self.dataset_name}...")
        common_data_loader = DataLoader({'dataset_name': self.dataset_name})
        self._max_k = min(min(common_data_loader.n_users, common_data_loader.n_items) - 1, 2000)

        for seed in self.seeds:
            # [CRITICAL] Clear any cached sparse/dense matrices before starting a new seed's trials
            clear_sparse_cache()
            
            print(f"\n>>> Starting HPO ({self.mode.upper()}) for Seed: {seed}")
            seed_dir = os.path.join(self.hpo_root, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            
            if self.mode == 'grid':
                sampler = optuna.samplers.GridSampler(self.get_search_space())
                actual_n_trials = None
            else:
                sampler = optuna.samplers.TPESampler(seed=seed)
                actual_n_trials = n_trials

            study = optuna.create_study(direction='maximize' if self.maximize else 'minimize', sampler=sampler)
            study.optimize(lambda t: self.objective(t, seed, data_loader=common_data_loader), n_trials=actual_n_trials)
            
            self.save_results(study, seed_dir)
            all_best_val_scores.append(study.best_value)
            all_best_params.append(study.best_params)
            
            with open(os.path.join(seed_dir, 'best_params.json'), 'w') as f:
                json.dump(study.best_params, f, indent=4)

            # Evaluate BEST on Test Set
            best_model_cfg = copy.deepcopy(self.model_cfg)
            for name, val in study.best_params.items():
                keys = (name if '.' in name else f"model.{name}").split('.')
                d = best_model_cfg
                for k in keys[:-1]: d = d.setdefault(k, {})
                d[keys[-1]] = val
            
            test_metrics = self.run_func(self.dataset_name, best_model_cfg, 
                                        output_path=os.path.join(seed_dir, 'best_test_run'),
                                        hpo_mode=False,
                                        data_loader=common_data_loader)
            all_seed_results.append(test_metrics)

        summary = self.report_final_results(all_seed_results, all_best_val_scores, all_best_params)
        return summary

    def report_final_results(self, test_results, best_val_scores, best_params_list):
        if not test_results: return {}
        all_keys = sorted(list(test_results[0].keys()))
        summary = {}
        print(f"\n{'#'*60}\n FINAL HPO REPORT ({len(self.seeds)} Seeds) \n{'#'*60}")
        
        # 1. Best Validation Score
        val_mean, val_std = np.mean(best_val_scores), np.std(best_val_scores)
        val_key = f"Best_VAL_{self.metric}"
        summary[f"{val_key}_mean"], summary[f"{val_key}_std"] = float(val_mean), float(val_std)
        print(f"{val_key:<20}: {val_mean:.4f} ± {val_std:.4f} (Selection Metric)")
        
        # 2. Test Results
        for key in all_keys:
            vals = [res[key] for res in test_results if key in res]
            mean, std = np.mean(vals), np.std(vals)
            summary[f"{key}_mean"], summary[f"{key}_std"] = float(mean), float(std)
            print(f"{key:<20}: {mean:.4f} ± {std:.4f}")
            
        # 3. Include best params in summary BEFORE saving
        summary['best_params_per_seed'] = best_params_list

        summary_path = os.path.join(self.hpo_root, 'final_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        return summary
