from .ease import EASE
from .mf import MF
from .puresvd import PureSVD
from .gf_cf import GF_CF
from .ials import iALS
from .lightgcn import LightGCN
from .ips_lae import IPS_LAE
from .lira import LIRA
from .dlae import DLAE
from .ipswiener import IPSWiener
from .aspire import Aspire
from .bspm import BSPM
from .turbocf import TurboCF
from .lr_chebyshev_cf import LR_Chebyshev_CF
from .mnar_lae import MNAR_LAE
from .constrained_mcar_ease import ConstrainedMCAREASE
from .alpha_ease import AlphaEASE

MODEL_REGISTRY = {
    'ease': EASE,
    'mf': MF,
    'puresvd': PureSVD,
    'gf_cf': GF_CF,
    'ials': iALS,
    'lightgcn': LightGCN,
    'ips_lae': IPS_LAE,
    'lira': LIRA,
    'dlae': DLAE,
    'ipswiener': IPSWiener,
    'aspire': Aspire,
    'bspm': BSPM,
    'turbocf': TurboCF,
    'lr_chebyshev_cf': LR_Chebyshev_CF,
    'mnar_lae': MNAR_LAE,
    'constrained_mcar_ease': ConstrainedMCAREASE,
    'alpha_ease': AlphaEASE
}

def get_model(model_name, config, data_loader):
    # 하이픈(-)과 언더바(_)를 동일하게 취급하여 매칭 유연성 확보
    norm_name = model_name.lower().replace('-', '_')
    if norm_name not in MODEL_REGISTRY:
        # puresvd 같은 경우 puresvd, pure_svd 둘 다 대응 가능하도록 추가 확인
        if norm_name == 'pure_svd' and 'puresvd' in MODEL_REGISTRY:
            norm_name = 'puresvd'
        elif norm_name == 'puresvd' and 'pure_svd' in MODEL_REGISTRY:
            norm_name = 'pure_svd'
        
        if norm_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name} (normalized: {norm_name}). Available models: {list(MODEL_REGISTRY.keys())}")
            
    return MODEL_REGISTRY[norm_name](config, data_loader)

def register_model(name, model_class):
    MODEL_REGISTRY[name] = model_class
