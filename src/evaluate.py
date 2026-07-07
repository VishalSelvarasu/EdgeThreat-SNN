import os
import json
import time
import argparse
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from encode_spikes import rate_encode
from model_baseline import BaselineMLP
from model_snn import EdgeThreatSNN

TIME_STEPS = 25


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def compute_far(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def compute_metrics(y_true, preds, probs):
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "far": float(compute_far(y_true, preds))
    }


def main(args):
    test_df = pd.read_csv(args.test)

    X_test = test_df.drop(columns=["label"]).values
    y_test = test_df["label"].values

    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    baseline = BaselineMLP(input_dim=X_test.shape[1])
    baseline.load_state_dict(torch.load(args.baseline_model, map_location="cpu"))
    baseline.eval()

    with torch.no_grad():
        t0 = time.perf_counter()
        baseline_logits = baseline(x_test_tensor)
        baseline_latency = (time.perf_counter() - t0) * 1000
        baseline_probs = torch.softmax(baseline_logits, dim=1)[:, 1].cpu().numpy()
        baseline_preds = baseline_logits.argmax(dim=1).cpu().numpy()

    snn = EdgeThreatSNN(input_dim=X_test.shape[1])
    snn.load_state_dict(torch.load(args.snn_model, map_location="cpu"))
    snn.eval()

    x_test_spikes = torch.tensor(rate_encode(X_test, TIME_STEPS), dtype=torch.float32)

    with torch.no_grad():
        t1 = time.perf_counter()
        mem_rec = snn(x_test_spikes)
        snn_latency = (time.perf_counter() - t1) * 1000
        snn_logits = mem_rec.sum(dim=0)
        snn_probs = torch.softmax(snn_logits, dim=1)[:, 1].cpu().numpy()
        snn_preds = snn_logits.argmax(dim=1).cpu().numpy()

    results = {
        "baseline_mlp": compute_metrics(y_test, baseline_preds, baseline_probs),
        "edge_threat_snn": compute_metrics(y_test, snn_preds, snn_probs)
    }

    results["baseline_mlp"]["latency_ms"] = float(baseline_latency)
    results["edge_threat_snn"]["latency_ms"] = float(snn_latency)
    results["edge_threat_snn"]["avg_spike_activity"] = float((x_test_spikes.numpy() > 0).mean())

    ensure_dir(os.path.dirname(args.output))
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="data/processed/test_processed.csv")
    parser.add_argument("--baseline_model", default="results/saved_models/baseline_mlp.pt")
    parser.add_argument("--snn_model", default="results/saved_models/edge_threat_snn.pt")
    parser.add_argument("--output", default="results/tables/metrics.json")
    args = parser.parse_args()
    main(args)