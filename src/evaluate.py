import json

import numpy as np
import pandas as pd

import config
from data import load_dataset
from models import DISPLAY_NAMES
from train import ALL_KINDS, seeds_for
from utils import (binary_metrics, fmt, load_any, model_path, predict_scores,
                   ranking_metrics, threshold_for_far)

FAMILIES = ["DoS", "Probe", "R2L", "U2R"]
MAIN_COLUMNS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("far", "FAR"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("recall_at_far_1", "Recall @ 1% FAR"),
    ("recall_at_far_5", "Recall @ 5% FAR"),
]


def evaluate_run(kind, seed, data):
    model = load_any(kind, seed)
    row = {"kind": kind, "seed": seed}
    if kind == "snn":
        model.record_stats = True
        model.reset_stats()
    score = predict_scores(
        model, data.X_test, draws=config.EVAL_SPIKE_DRAWS, seed=seed)
    if kind == "snn":
        s, steps = model.stats, model.stats["samples"] * model.time_steps
        row["input_spike_rate"] = s["input_events"] / (steps * data.n_features)
        row["hidden_spike_rate"] = s["hidden_spikes"] / \
            (steps * model.fc1.out_features)
        row["input_events_per_sample"] = s["input_events"] / s["samples"]
        row["hidden_spikes_per_sample"] = s["hidden_spikes"] / s["samples"]
        model.record_stats = False
    val_score = predict_scores(
        model, data.X_val, draws=config.EVAL_SPIKE_DRAWS, seed=seed + 500)

    config.PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.PRED_DIR / f"{kind}_s{seed}.npz", test=score, val=val_score)

    # default threshold = probability 0.5
    row.update(binary_metrics(data.y_test, score))
    row.update(ranking_metrics(data.y_test, score))

    # chosen on validation only
    thr = threshold_for_far(data.y_val, val_score, config.VAL_FAR_TARGET)
    cal = binary_metrics(data.y_test, score, thr)
    row["cal_recall"], row["cal_far"] = cal["recall"], cal["far"]

    # Per-family recall with every model held to the same 5% false-alarm rate.
    pred = score >= threshold_for_far(data.y_test, score, config.FAMILY_FAR)
    for fam in FAMILIES:
        mask = data.test_family == fam
        row[f"recall_{fam}"] = float(
            pred[mask].mean()) if mask.any() else np.nan
    seen = (data.y_test == 1) & ~data.test_novel
    row["recall_seen_types"] = float(pred[seen].mean())
    row["recall_unseen_types"] = float(
        pred[data.test_novel].mean()) if data.test_novel.any() else np.nan
    return row


def aggregate(runs, columns):
    out = {}
    for kind, group in runs.groupby("kind", sort=False):
        stats = {"n": int(len(group))}
        for col in columns:
            if col in group and group[col].notna().any():
                v = group[col].astype(float)
                stats[col] = {"mean": float(v.mean()), "std": float(
                    v.std(ddof=1)) if len(v) > 1 else None}
        out[kind] = stats
    return out


def markdown_table(summary, columns):
    lines = ["| Model | Seeds | " + " | ".join(h for _, h in columns) + " |",
             "| --- | ---: | " + " | ".join("---:" for _ in columns) + " |"]
    for kind, stats in summary.items():
        cells = ["–" if stats.get(c) is None else fmt(
            stats[c]["mean"], stats[c]["std"]) for c, _ in columns]
        lines.append(
            f"| {DISPLAY_NAMES[kind]} | {stats['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    data = load_dataset(log_scale=config.LOG_SCALE)
    config.PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.PRED_DIR / "y_test.npz", y=data.y_test)

    rows = []
    for kind in ALL_KINDS:
        for seed in seeds_for(kind, config.SEEDS):
            if not model_path(kind, seed).exists():
                print(f"missing {model_path(kind, seed).name}, skipped")
                continue
            print(f"evaluating {kind} seed {seed}")
            rows.append(evaluate_run(kind, seed, data))
    runs = pd.DataFrame(rows)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.TABLE_DIR / "runs.csv", index=False)

    extra = ["cal_recall", "cal_far", "input_spike_rate", "hidden_spike_rate",
             "input_events_per_sample", "hidden_spikes_per_sample"]
    fam_cols = [f"recall_{f}" for f in FAMILIES] + \
        ["recall_seen_types", "recall_unseen_types"]
    summary = aggregate(runs, [c for c, _ in MAIN_COLUMNS] + extra)
    families = aggregate(runs, fam_cols)

    flat = []
    for kind, stats in summary.items():
        rec = {"model": DISPLAY_NAMES[kind], "n": stats["n"]}
        for col, s in stats.items():
            if col != "n":
                rec[col], rec[col + "_std"] = s["mean"], s["std"]
        flat.append(rec)
    pd.DataFrame(flat).to_csv(config.TABLE_DIR / "summary.csv", index=False)

    # The tables contain non-ASCII characters, so the encoding is set explicitly
    # (Windows would otherwise write them in its legacy code page).
    (config.TABLE_DIR / "summary.md").write_text(
        "## Test results (KDDTest+; threshold metrics at probability 0.5)\n\n"
        + markdown_table(summary, MAIN_COLUMNS)
        + "\n## Threshold calibrated on validation data (1% FAR on validation)\n\n"
        + markdown_table(summary, [("cal_recall",
                         "Test recall"), ("cal_far", "Test FAR")]),
        encoding="utf-8")

    def n(mask):
        return f"{int(mask.sum()):,}"

    headers = [(f"recall_{f}", f"{f} (n={n(data.test_family == f)})")
               for f in FAMILIES]
    headers += [("recall_seen_types", f"Seen types (n={n((data.y_test == 1) & ~data.test_novel)})"),
                ("recall_unseen_types", f"Unseen types (n={n(data.test_novel)})")]
    (config.TABLE_DIR / "families.md").write_text(
        "## Recall by attack family, every model at 5% FAR\n\n"
        + markdown_table(families, headers),
        encoding="utf-8")
    (config.TABLE_DIR / "results.json").write_text(
        json.dumps({"summary": summary, "families": families}, indent=1), encoding="utf-8")
    print((config.TABLE_DIR / "summary.md").read_text(encoding="utf-8"))
    print((config.TABLE_DIR / "families.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
