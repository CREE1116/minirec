import os
import yaml
import copy

def deep_merge(base, override):
    """base에 override를 deep merge. override 값이 우선."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_yaml(path):
    """YAML 파일을 로드합니다. 파일이 없으면 에러를 발생시킵니다."""
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML config file not found at: {os.path.abspath(path)}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def merge_all_configs(dataset_config, model_config, eval_config_path=None):
    """마스터 eval (기본값) -> dataset -> model 순서로 deep merge."""
    if eval_config_path is None:
        # minirec/configs/evaluation.yaml 또는 전역 configs/evaluation.yaml 탐색
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # src/utils -> src -> minirec -> configs
        potential_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'configs', 'evaluation.yaml'))
        if os.path.exists(potential_path):
            eval_config_path = potential_path
        else:
            # Fallback to root configs
            root_eval_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'configs', 'evaluation.yaml'))
            if os.path.exists(root_eval_path):
                eval_config_path = root_eval_path
            
    eval_config = load_yaml(eval_config_path)
    
    # eval_config의 내용이 'evaluation' 키 아래에 있지 않다면 감싸줌 (Trainer 등에서 config['evaluation']을 참조함)
    if eval_config and 'evaluation' not in eval_config:
        eval_config = {'evaluation': eval_config}
    
    # 1. eval + dataset
    config = deep_merge(eval_config, dataset_config)
    # 2. (eval+dataset) + model
    config = deep_merge(config, model_config)
    
    return config
