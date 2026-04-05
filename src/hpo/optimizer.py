import optuna
import copy
import os
import json

class BayesianOptimizer:
    def __init__(self, run_func, dataset_cfg, model_cfg, hpo_cfg):
        self.run_func = run_func
        self.dataset_cfg = dataset_cfg
        self.model_cfg = model_cfg
        self.hpo_cfg = hpo_cfg
        self.metric = hpo_cfg.get('metric', 'NDCG@10')
        self.maximize = hpo_cfg.get('direction', 'max') == 'max'

    def objective(self, trial):
        model_cfg = copy.deepcopy(self.model_cfg)
        params = self.hpo_cfg.get('params', [])
        
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
            
            # Nested set (e.g. "model.lr")
            keys = name.split('.')
            d = model_cfg
            for k in keys[:-1]: d = d.setdefault(k, {})
            d[keys[-1]] = val
            
        metrics = self.run_func(self.dataset_cfg, model_cfg, output_path=None, hpo_mode=True)
        return metrics.get(self.metric, 0.0)

    def search(self, n_trials=20):
        study = optuna.create_study(direction='maximize' if self.maximize else 'minimize')
        study.optimize(self.objective, n_trials=n_trials)
        print(f"Best params: {study.best_params}")
        return study.best_params
