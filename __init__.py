import os
import torch
import random
import numpy as np

# src 폴더 내부의 핵심 로직들을 노출
from src.utils.config import merge_all_configs, load_yaml
from src.data.loader import DataLoader
from src.models import get_model
from src.trainer import Trainer
from src.hpo.optimizer import BayesianOptimizer

from src.utils.seed import set_seed

def run(dataset_cfg, model_cfg, output_path=None, hpo_mode=False):
    """
    단일 실험 실행 함수.
    """
    # 1. Load/Merge Configs
    if isinstance(dataset_cfg, str): dataset_cfg = load_yaml(dataset_cfg)
    if isinstance(model_cfg, str): model_cfg = load_yaml(model_cfg)
    
    # 데이터셋 설정에서 시드 확인 (없으면 None으로 넘겨서 랜덤 생성 유도)
    config_seed = dataset_cfg.get('seed')
    current_seed = set_seed(config_seed)
    
    # 2. Setup Config
    config = merge_all_configs(dataset_cfg, model_cfg)
    config['seed'] = current_seed
    
    # 기본 출력 경로 설정
    if output_path is None:
        output_path = config.get('output_path_override', 'output')
    config['output_path_override'] = output_path
    if hpo_mode: config['hpo_mode'] = True
    
    # 3. Load Data & Model
    data_loader = DataLoader(config)
    
    # 4. Get Model & Device
    model_name = config.get('model_name')
    if not model_name and 'model' in config:
        model_name = config['model'].get('model_name', config['model'].get('name', 'MF'))
    else:
        model_name = model_name or 'MF'
        
    model = get_model(model_name, config, data_loader)
    
    # 5. Train & Evaluate
    trainer = Trainer(config, model, data_loader)
    return trainer.run()

def hporun(dataset_cfg, model_cfg, hpo_cfg, n_trials=20):
    """
    하이퍼파라미터 탐색 실행 함수 (멀티시드 지원).
    mode: 'bayesian' (기본값) 또는 'grid'
    """
    # BayesianOptimizer가 hpo_cfg를 통해 멀티시드 및 탐색 모드(bayesian/grid)를 처리함
    optimizer = BayesianOptimizer(run, dataset_cfg, model_cfg, hpo_cfg)

    # grid 모드일 경우 n_trials를 무시하거나 search 내부에서 처리하도록 함
    summary = optimizer.search(n_trials=n_trials)

    print(f"HPO ({hpo_cfg.get('mode', 'bayesian')}) Finished.")
    return summary

