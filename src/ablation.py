import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import config
from data import load_dataset
from train import train_neural
from utils import binary_metrics, fmt, predict_scores, ranking_metrics

ABL_DIR = config.RESULTS_DIR / "ablation"
DEFAULTS = {"time_steps": config.TIME_STEPS, "beta": config.BETA,
            "encoding": config.ENCODING, "log_scale": config.LOG_SCALE}
LABELS = {"time_steps": "Time steps", "beta": "Membrane decay (beta)",
          "encoding": "Input encoding", "log_scale": "log1p before scaling"}

_data_cache = {}


def get_data(log_scale):
    if log_scale not in _data_cache:
        _data_cache[log_scale] = load_dataset(log_scale=log_scale)
    return _data_cache[log_scale]


def jobs():
    out = []
    for factor, values in config.ABLATION_GRID.items():
        for value in values:
            if value == DEFAULTS[factor]:
                continue
            for seed in config.ABLATION_SEEDS:
                out.append((factor, value, seed))
    out += [("default", None, s) for s in config.ABLATION_SEEDS]
    return out


def job_name(factor, value, seed):
    return f"default_s{seed}" if factor == "default" else f"{factor}={value}_s{seed}"


def evaluate(model, data, seed, val_loss):
    model.record_stats = True
    model.reset_stats()
    p_test = predict_scores(
        model, data.X_test, draws=config.EVAL_SPIKE_DRAWS, seed=seed)
    s = model.stats
    hidden_rate = s["hidden_spikes"] / \
        (s["samples"] * model.time_steps * model.fc1.out_features)
    model.record_stats = False
    p_val = predict_scores(
        model, data.X_val, draws=config.EVAL_SPIKE_DRAWS, seed=seed + 500)
    test = {**binary_metrics(data.y_test, p_test), **
            ranking_metrics(data.y_test, p_test)}
    return {
        "val_loss": val_loss,
        "val_roc_auc": ranking_metrics(data.y_val, p_val)["roc_auc"],
        "test_roc_auc": test["roc_auc"],
        "test_recall_at_far_1": test["recall_at_far_1"],
        "test_recall_at_far_5": test["recall_at_far_5"],
        "test_f1": test["f1"],
        "hidden_spike_rate": hidden_rate,
    }


def run_job(factor, value, seed):
    settings = dict(DEFAULTS)
    if factor != "default":
        settings[factor] = value
    data = get_data(settings["log_scale"])
    snn_kwargs = {k: settings[k] for k in ("time_steps", "beta", "encoding")}

    model, info = train_neural("snn", seed, data, snn_kwargs=snn_kwargs,
                               max_epochs=config.ABLATION_MAX_EPOCHS, log=lambda m: None)
    result = {"factor": factor, "value": value, "seed": seed, "settings": settings,
              "best_epoch": info["best_epoch"], "train_seconds": info["train_seconds"],
              **evaluate(model, data, seed, info["best_val_loss"])}
    return result


def run_all():
    ABL_DIR.mkdir(parents=True, exist_ok=True)
    for factor, value, seed in jobs():
        name = job_name(factor, value, seed)
        out, lock = ABL_DIR / f"{name}.json", ABL_DIR / f"{name}.lock"
        if out.exists():
            continue
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            continue  # another process is on it
        try:
            print(f"running {name}", flush=True)
            result = run_job(factor, value, seed)
            out.write_text(json.dumps(result, indent=1))
            print(f"  done: test ROC-AUC {result['test_roc_auc']:.3f}, "
                  f"recall@5%FAR {result['test_recall_at_far_5']:.3f}", flush=True)
        finally:
            lock.unlink(missing_ok=True)


def report():
    rows = [json.loads(p.read_text()) for p in sorted(ABL_DIR.glob("*.json"))]
    if not rows:
        raise SystemExit("no ablation results yet")
    records = []
    for factor, values in config.ABLATION_GRID.items():
        for value in values:
            if value == DEFAULTS[factor]:
                group = [r for r in rows if r["factor"] == "default"]
            else:
                group = [r for r in rows if r["factor"]
                         == factor and r["value"] == value]
            if not group:
                continue
            rec = {"factor": factor, "value": value, "is_default": value == DEFAULTS[factor],
                   "n": len(group)}
            for key in ("val_roc_auc", "test_roc_auc", "test_recall_at_far_1",
                        "test_recall_at_far_5", "test_f1", "hidden_spike_rate", "best_epoch"):
                v = np.array([r[key] for r in group], dtype=float)
                rec[key] = v.mean()
                rec[key + "_std"] = v.std(ddof=1) if len(v) > 1 else np.nan
            records.append(rec)
    df = pd.DataFrame(records)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.TABLE_DIR / "ablation.csv", index=False)

    lines = ["| Setting | Value | Val ROC-AUC | Test ROC-AUC | Test recall @ 1% FAR | "
             "Test recall @ 5% FAR | Hidden spike rate | Seeds |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in records:
        value = f"**{r['value']}** (default)" if r["is_default"] else str(r["value"])
        lines.append(
            f"| {LABELS[r['factor']]} | {value} | {fmt(r['val_roc_auc'], r['val_roc_auc_std'], 4)} | "
            f"{fmt(r['test_roc_auc'], r['test_roc_auc_std'])} | "
            f"{fmt(r['test_recall_at_far_1'], r['test_recall_at_far_1_std'])} | "
            f"{fmt(r['test_recall_at_far_5'], r['test_recall_at_far_5_std'])} | "
            f"{100 * r['hidden_spike_rate']:.1f}% | {r['n']} |")
    (config.TABLE_DIR / "ablation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    report() if args.report else run_all()
