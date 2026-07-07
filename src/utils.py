import os
import time
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def label_to_binary(series):
    return series.apply(lambda x: 0 if str(x).lower() == "normal" else 1).astype(int)


def compute_metrics(y_true, y_pred, y_score=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "far": ((np.array(y_pred) == 1) & (np.array(y_true) == 0)).sum() / max((np.array(y_true) == 0).sum(), 1),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    return metrics


def save_artifact(obj, path):
    ensure_dir(os.path.dirname(path))
    joblib.dump(obj, path)


def now_ms():
    return time.perf_counter() * 1000
