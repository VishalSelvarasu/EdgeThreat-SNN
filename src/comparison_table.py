import json
import pandas as pd

with open("results/tables/baseline_metrics.json", "r") as f:
    baseline = json.load(f)

with open("results/tables/snn_metrics.json", "r") as f:
    snn = json.load(f)

rows = [
    {
        "model": "Baseline MLP",
        "accuracy": baseline.get("accuracy"),
        "precision": baseline.get("precision"),
        "recall": baseline.get("recall"),
        "f1": baseline.get("f1"),
        "roc_auc": baseline.get("roc_auc"),
        "far": baseline.get("far"),
        "latency_ms": baseline.get("latency_ms", None),
        "avg_spike_activity": None,
    },
    {
        "model": "EdgeThreat-SNN",
        "accuracy": snn.get("accuracy"),
        "precision": snn.get("precision"),
        "recall": snn.get("recall"),
        "f1": snn.get("f1"),
        "roc_auc": snn.get("roc_auc"),
        "far": snn.get("far"),
        "latency_ms": snn.get("latency_ms"),
        "avg_spike_activity": snn.get("avg_spike_activity"),
    }
]

df = pd.DataFrame(rows)
df.to_csv("results/tables/model_comparison.csv", index=False)
print(df)
print("\nSaved to results/tables/model_comparison.csv")