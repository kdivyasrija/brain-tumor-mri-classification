"""Reproducibility helpers."""
import os, random, platform
import numpy as np


def set_seed(seed=42):
    """Seed Python, NumPy, PyTorch and CUDA. Deterministic cuDNN is requested (can be slightly slower)."""
    import torch
    random.seed(seed); np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def env_info(seed=42):
    import torch, torchvision
    return dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
                cuda_available=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, seed=seed)
