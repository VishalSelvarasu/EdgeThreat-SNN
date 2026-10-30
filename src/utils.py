import random

import joblib
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

import config
from models import NEURAL_KINDS, EdgeThreatSNN, build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def model_path(kind, seed):
    ext = "pt" if kind in NEURAL_KINDS else "joblib"
    return config.MODEL_DIR / f"{kind}_s{seed}.{ext}"


def save_neural(model, kind, input_dim, info, path):
    """Save everything needed to rebuild the exact model, not just the weights."""
    path.parent.mkdir(parents=True, exist_ok=True)
    snn_kwargs = None
    if isinstance(model, EdgeThreatSNN):
        snn_kwargs = {"time_steps": model.time_steps, "encoding": model.encoding,
                      "beta": float(model.lif1.beta)}
    torch.save({"kind": kind, "input_dim": input_dim, "snn": snn_kwargs,
                "state_dict": model.state_dict(), "info": info}, path)


def load_neural(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["kind"], ckpt["input_dim"], **(ckpt["snn"] or {}))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def load_any(kind, seed):
    path = model_path(kind, seed)
    return load_neural(path)[0] if kind in NEURAL_KINDS else joblib.load(path)


@torch.no_grad()
def predict_scores(model, X, draws=1, seed=0, batch=4096):
    """Attack score as log-odds (0 = probability 0.5). Float32 softmax probabilities
    saturate at exactly 1.0, which creates ties and breaks ROC metrics at low
    false-alarm rates. A rate-coded SNN's score is averaged over `draws` encodings."""
    if not isinstance(model, torch.nn.Module):
        if hasattr(model, "decision_function"):
            return model.decision_function(X).astype(np.float64)
        p = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    model.eval()
    n_draws = draws if isinstance(
        model, EdgeThreatSNN) and model.encoding == "rate" else 1
    Xt = torch.as_tensor(X, dtype=torch.float32)
    out = np.zeros(len(X), dtype=np.float64)
    for d in range(n_draws):
        gen = torch.Generator().manual_seed(10_007 * (seed + 1) + d)
        for i in range(0, len(Xt), batch):
            logits = model(Xt[i:i + batch], generator=gen).double()
            out[i:i + batch] += (logits[:, 1] - logits[:, 0]).numpy()
    return out / n_draws


def to_probability(score):
    return 1.0 / (1.0 + np.exp(-np.asarray(score, dtype=np.float64)))


def binary_metrics(y, score, threshold=0.0):
    pred = (score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "far": float(pred[y == 0].mean()),
    }


def recall_at_far(y, score, far):
    """Highest recall reachable with a false-alarm rate of at most `far`."""
    fpr, tpr, _ = roc_curve(y, score)
    return float(tpr[fpr <= far + 1e-12].max())


def threshold_for_far(y, score, far):
    """Lowest threshold that flags at most `far` of the normal rows."""
    normal = np.sort(score[y == 0])
    allowed = int(np.floor(far * len(normal)))
    return float(np.nextafter(normal[len(normal) - allowed - 1], np.inf))


def ranking_metrics(y, score):
    out = {"roc_auc": float(roc_auc_score(y, score)),
           "pr_auc": float(average_precision_score(y, score))}
    for f in config.FAR_TARGETS:
        out[f"recall_at_far_{int(f * 100)}"] = recall_at_far(y, score, f)
    return out


def fmt(mean, std=None, digits=3):
    if std is None or np.isnan(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"
