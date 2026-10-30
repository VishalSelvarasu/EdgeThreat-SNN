from train import seeds_for
from models import DISPLAY_NAMES
import config
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

import matplotlib

matplotlib.use("Agg")


STYLE = {
    "mlp": {"color": "#1f5fbf", "ls": "-"},
    "mlp_small": {"color": "#6aa0e8", "ls": "-"},
    "snn": {"color": "#d9711c", "ls": "-"},
    "rf": {"color": "#7a7a7a", "ls": "--"},
    "logreg": {"color": "#b0b0b0", "ls": ":"},
}
GRID = np.linspace(0, 0.20, 401)


def load_probs(kind):
    out = []
    for seed in seeds_for(kind, config.SEEDS):
        path = config.PRED_DIR / f"{kind}_s{seed}.npz"
        if path.exists():
            out.append(np.load(path)["test"])
    return out


def mean_roc(y, probs):
    curves = []
    for p in probs:
        fpr, tpr, _ = roc_curve(y, p)
        # step-wise: best recall reachable at each false-alarm rate on the grid
        curves.append([tpr[fpr <= g + 1e-12].max() for g in GRID])
    curves = np.array(curves)
    return curves.mean(0), curves.std(0)


def plot_roc(y, summary):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for kind in ("rf", "mlp", "mlp_small", "snn"):
        probs = load_probs(kind)
        if not probs:
            continue
        mean, std = mean_roc(y, probs)
        st = STYLE[kind]
        ax.plot(GRID * 100, mean * 100, color=st["color"], ls=st["ls"], lw=2,
                label=f"{DISPLAY_NAMES[kind]} (n={len(probs)})")
        if kind in ("mlp", "snn"):
            ax.fill_between(GRID * 100, (mean - std) * 100, (mean + std)
                            * 100, color=st["color"], alpha=0.15, lw=0)
        if kind in ("mlp", "mlp_small", "snn"):
            # each seed's own default-threshold operating point (an averaged point would not lie on any curve)
            runs = pd.read_csv(config.TABLE_DIR / "runs.csv")
            r = runs[runs["kind"] == kind]
            ax.plot(r["far"] * 100, r["recall"] * 100, "o", ms=5, color=st["color"],
                    mec="white", mew=0.8, zorder=5)
    for far in config.FAR_TARGETS:
        ax.axvline(far * 100, color="#999999", lw=0.8, ls=":")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 100)
    ax.set_xlabel("False-alarm rate (% of normal connections flagged)")
    ax.set_ylabel("Attacks detected (recall, %)")
    ax.set_title("KDDTest+: recall at each false-alarm rate (mean ± 1 sd over seeds)\n"
                 "dots = each seed at the default 0.5 threshold; dotted lines = 1% and 5% budgets", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "roc_comparison.png", dpi=160)
    plt.close(fig)


def plot_families(families):
    groups = ["recall_DoS", "recall_Probe", "recall_R2L",
              "recall_U2R", "recall_unseen_types"]
    labels = ["DoS", "Probe", "R2L", "U2R", "Unseen types"]
    kinds = [k for k in ("mlp", "mlp_small", "snn", "rf") if k in families]
    width = 0.8 / len(kinds)
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, kind in enumerate(kinds):
        means = [families[kind][g]["mean"] * 100 for g in groups]
        errs = [(families[kind][g]["std"] or 0) * 100 for g in groups]
        ax.bar(x + (i - (len(kinds) - 1) / 2) * width, means, width, yerr=errs, capsize=2,
               color=STYLE[kind]["color"], label=DISPLAY_NAMES[kind])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Recall at 5% false-alarm rate (%)")
    ax.set_title("Recall by attack family on KDDTest+, every model at a 5% false-alarm rate\n"
                 "(mean ± 1 sd over seeds)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "family_recall.png", dpi=160)
    plt.close(fig)


def plot_ablation():
    path = config.TABLE_DIR / "ablation.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    factors = [f for f in config.ABLATION_GRID if f in set(df["factor"])]
    titles = {"time_steps": "Time steps", "beta": "Membrane decay β",
              "encoding": "Input encoding", "log_scale": "log1p before scaling"}
    fig, axes = plt.subplots(1, len(factors), figsize=(
        3.2 * len(factors), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, factor in zip(axes, factors):
        sub = df[df["factor"] == factor]
        x = np.arange(len(sub))
        for col, label, color, off in (("test_roc_auc", "Test ROC-AUC", "#1f5fbf", -0.1),
                                       ("test_recall_at_far_5", "Test recall @ 5% FAR", "#d9711c", 0.1)):
            ax.errorbar(x + off, sub[col], yerr=sub[col + "_std"].fillna(0), fmt="o", color=color,
                        capsize=3, label=label)
        ax.set_xticks(x, [str(v) + (" *" if d else "")
                      for v, d in zip(sub["value"], sub["is_default"])])
        ax.set_title(titles[factor], fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score (mean ± 1 sd, 3 seeds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=2, fontsize=9, frameon=False)
    fig.suptitle(
        "SNN sensitivity, one setting changed at a time (* = default; 10-epoch budget)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(config.FIG_DIR / "ablation.png", dpi=160)
    plt.close(fig)


def main():
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    y = np.load(config.PRED_DIR / "y_test.npz")["y"]
    results = json.loads((config.TABLE_DIR / "results.json").read_text())
    plot_roc(y, results["summary"])
    plot_families(results["families"])
    plot_ablation()
    print(f"figures written to {config.FIG_DIR}")


if __name__ == "__main__":
    main()
