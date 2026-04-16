from .ease import EASE
from .mf import MF
from .puresvd import PureSVD
from .gf_cf import GF_CF
from .ials import iALS
from .lightgcn import LightGCN
from .ips_lae import IPS_LAE
from .lae import LAE
from .dlae import DLAE
from .bspm import BSPM
from .turbocf import TurboCF
from .dan import EASE_DAN, DLAE_DAN
from .rlae import RLAE, RDLAE
from .causal_aspire import CausalAspire
from .fixed_aspire import FixedAspire
from .mf_ips import MF_IPS
from .dr_jl import DR_JL
from .co_occurrence import CoOccurrence
from .pmi_aspire import PMIAspire   
from .pmi_lae import PMILAE

MODEL_REGISTRY = {
    'ease': EASE,
    'mf': MF,
    'puresvd': PureSVD,
    'gf_cf': GF_CF,
    'ials': iALS,
    'lightgcn': LightGCN,
    'ips_lae': IPS_LAE,
    'lae': LAE,
    'dlae': DLAE,
    'bspm': BSPM,
    'turbocf': TurboCF,
    'ease_dan': EASE_DAN,
    'dlae_dan': DLAE_DAN,
    'rlae': RLAE,
    'rdlae': RDLAE,
    'causal_aspire': CausalAspire,
    'fixed_aspire': FixedAspire,
    'mf_ips': MF_IPS,
    'dr_jl': DR_JL,
    'co_occurrence': CoOccurrence,
    'pmi_lae': PMILAE,
    'pmi_aspire': PMIAspire,
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


