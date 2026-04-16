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

def run(dataset_name, model_cfg, output_path=None, hpo_mode=False, use_test_for_hpo=False):
    """
    단일 실험 실행 함수.
    dataset_name: 'ml-100k', 'steam' 등 (data/preprocessed/ 하위 폴더명)
    """
    # 1. Load Configs
    if isinstance(model_cfg, str): model_cfg = load_yaml(model_cfg)
    eval_cfg = load_yaml('configs/evaluation.yaml')
    
    # 시드는 평가 설정에서 가져옴 (없으면 None)
    config_seed = eval_cfg.get('seed')
    current_seed = set_seed(config_seed)
    
    # 2. Setup Config
    # dataset_cfg는 이제 이름만 포함하는 딕셔너리로 취급
    dataset_cfg = {'dataset_name': dataset_name}
    config = merge_all_configs(dataset_cfg, model_cfg)
    config['seed'] = current_seed
    
    # 기본 출력 경로 설정
    if output_path is None:
        output_path = config.get('output_path_override', 'output')
    config['output_path_override'] = output_path
    if hpo_mode: config['hpo_mode'] = True
    if use_test_for_hpo: config['use_test_for_hpo'] = True
    
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

def hporun(dataset_name, model_cfg, hpo_cfg, n_trials=20):
    """
    하이퍼파라미터 탐색 실행 함수 (멀티시드 지원).
    mode: 'bayesian' (기본값) 또는 'grid'
    """
    optimizer = BayesianOptimizer(run, dataset_name, model_cfg, hpo_cfg)
    summary = optimizer.search(n_trials=n_trials)

    print(f"HPO ({hpo_cfg.get('mode', 'bayesian')}) Finished.")
    return summary

