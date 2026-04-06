import torch
import numpy as np
import random
import os

def set_seed(seed=None):
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[Seed] Global seed set to: {seed}")
    return seed
