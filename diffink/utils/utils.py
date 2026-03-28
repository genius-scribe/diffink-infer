import os
import random
import yaml
import numpy as np
import torch
import torch.nn as nn


def load_config_from_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class ModelConfig:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[seed] set to {seed}")
