import torch
import numpy as np
import random
import os

def set_seed(seed=None):
    """
    전역 시드를 고정합니다. seed가 None이면 랜덤하게 생성합니다.
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[Seed] Global seed set to: {seed}")
    return seed
