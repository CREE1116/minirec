from .ease import EASE
from .mf import MF
from .pure_svd import PureSVD
from .gf_cf import GF_CF
from .ials import iALS
from .lightgcn import LightGCN
from .ips_lae import IPS_LAE
from .lira import LIRA

MODEL_REGISTRY = {
    'ease': EASE,
    'mf': MF,
    'pure_svd': PureSVD,
    'gf_cf': GF_CF,
    'ials': iALS,
    'lightgcn': LightGCN,
    'ips_lae': IPS_LAE,
    'lira': LIRA
}

def get_model(model_name, config, data_loader):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](config, data_loader)

def register_model(name, model_class):
    MODEL_REGISTRY[name] = model_class
