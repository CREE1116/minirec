import os
import torch
import random
import numpy as np

# src 폴더 내부의 핵심 로직들을 노출
from .src.utils.config import merge_all_configs, load_yaml
from .src.data.loader import DataLoader
from .src.models import get_model
from .src.trainer import Trainer
from .src.hpo.optimizer import BayesianOptimizer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run(dataset_cfg, model_cfg, output_path=None, hpo_mode=False):
    """
    단일 실험 실행 함수.
    dataset_cfg: dict 또는 yaml 경로
    model_cfg: dict 또는 yaml 경로
    output_path: 결과 저장 경로
    """
    # 1. Load/Merge Configs
    if isinstance(dataset_cfg, str): dataset_cfg = load_yaml(dataset_cfg)
    if isinstance(model_cfg, str): model_cfg = load_yaml(model_cfg)
    
    config = merge_all_configs(dataset_cfg, model_cfg)
    if output_path: config['output_path_override'] = output_path
    if hpo_mode: config['hpo_mode'] = True
    
    # 2. Setup Device & Seed
    set_seed(config.get('seed', 42))
    if config.get('device', 'auto') == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 3. Load Data & Model
    data_loader = DataLoader(config)
    model = get_model(config['model']['name'], config, data_loader)
    
    # 4. Train & Evaluate
    trainer = Trainer(config, model, data_loader)
    return trainer.run()

def hporun(dataset_cfg, model_cfg, hpo_cfg, n_trials=20):
    """
    하이퍼파라미터 탐색 실행 함수.
    hpo_cfg: { 'metric': 'NDCG@10', 'direction': 'max', 'params': [...] }
    """
    optimizer = BayesianOptimizer(run, dataset_cfg, model_cfg, hpo_cfg)
    best_params = optimizer.search(n_trials=n_trials)
    
    print(f"HPO Finished. Best Params: {best_params}")
    return best_params
