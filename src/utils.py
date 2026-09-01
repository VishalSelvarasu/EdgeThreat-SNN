import os
import random
import time
import numpy as np
import torch
from config import RANDOM_STATE
from sklearn.metrics import confusion_matrix


def set_all_seeds(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def now_ms():
    return time.perf_counter() * 1000


def compute_far(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0
