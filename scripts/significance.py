from pathlib import Path

import pandas as pd
from scipy.stats import ttest_ind

ROOT = Path(__file__).resolve().parents[1]
runs = pd.read_csv(ROOT / "results" / "tables" / "runs.csv")
PAIRS = [("snn", "mlp"), ("snn", "mlp_small"), ("rf", "snn")]
METRICS = [("roc_auc", "ROC-AUC"), ("recall_at_far_1", "Recall @ 1% FAR"),
           ("recall_at_far_5", "Recall @ 5% FAR"), ("f1", "F1 @ 0.5")]

lines = ["| Comparison | Metric | Difference (A − B) | p (Welch) |", "| --- | --- | ---: | ---: |"]
for a, b in PAIRS:
    for col, name in METRICS:
        x, y = runs.loc[runs.kind == a, col], runs.loc[runs.kind == b, col]
        p = ttest_ind(x, y, equal_var=False).pvalue
        shown = f"{p:.3f}" if p >= 0.001 else "< 0.001"
        lines.append(f"| {a} vs {b} | {name} | {x.mean() - y.mean():+.3f} | {shown} |")
out = ROOT / "results" / "tables" / "significance.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))