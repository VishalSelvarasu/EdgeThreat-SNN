import os
import json
import time
import argparse

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from encode_spikes import rate_encode
from model_baseline import BaselineMLP
from model_snn import EdgeThreatSNN
from config import TIME_STEPS, RANDOM_STATE
from utils import ensure_dir, set_all_seeds, compute_far


def main(args):
    set_all_seeds()

    # Load processed binary-classification test data.
    test_df = pd.read_csv(args.test)

    X_test = test_df.drop(columns=["label"]).values
    y_test = test_df["label"].values

    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Load and evaluate baseline MLP.
    baseline = BaselineMLP(input_dim=X_test.shape[1])
    baseline.load_state_dict(
        torch.load(args.baseline_model, map_location="cpu")
    )
    baseline.eval()

    with torch.no_grad():
        t0 = time.perf_counter()

        baseline_logits = baseline(x_test_tensor)

        baseline_latency = (time.perf_counter() - t0) * 1000
        baseline_probs = torch.softmax(baseline_logits, dim=1)[
            :, 1].cpu().numpy()
        baseline_preds = baseline_logits.argmax(dim=1).cpu().numpy()

    # Load and evaluate EdgeThreat-SNN.
    snn = EdgeThreatSNN(input_dim=X_test.shape[1])
    snn.load_state_dict(
        torch.load(args.snn_model, map_location="cpu")
    )
    snn.eval()

    x_test_spikes = torch.tensor(
        rate_encode(X_test, TIME_STEPS),
        dtype=torch.float32
    )

    with torch.no_grad():
        t1 = time.perf_counter()

        mem_rec = snn(x_test_spikes)

        snn_latency = (time.perf_counter() - t1) * 1000
        snn_logits = mem_rec.sum(dim=0)
        snn_probs = torch.softmax(snn_logits, dim=1)[:, 1].cpu().numpy()
        snn_preds = snn_logits.argmax(dim=1).cpu().numpy()

    # Save exact data from this evaluation run.
    # plot_results.py will use these files to create genuine ROC curves
    # and confusion matrices instead of manually entered values.
    predictions_dir = "results/predictions"
    ensure_dir(predictions_dir)

    np.save(os.path.join(predictions_dir, "y_true.npy"), y_test)
    np.save(os.path.join(predictions_dir, "baseline_preds.npy"), baseline_preds)
    np.save(os.path.join(predictions_dir, "baseline_probs.npy"), baseline_probs)
    np.save(os.path.join(predictions_dir, "snn_preds.npy"), snn_preds)
    np.save(os.path.join(predictions_dir, "snn_probs.npy"), snn_probs)

    results = {
        "baseline_mlp": compute_metrics(
            y_test,
            baseline_preds,
            baseline_probs
        ),
        "edge_threat_snn": compute_metrics(
            y_test,
            snn_preds,
            snn_probs
        ),
    }

    results["baseline_mlp"]["latency_ms"] = float(baseline_latency)
    results["edge_threat_snn"]["latency_ms"] = float(snn_latency)
    results["edge_threat_snn"]["avg_spike_activity"] = float(
        (x_test_spikes.numpy() > 0).mean()
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        ensure_dir(output_dir)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved prediction artifacts to: {predictions_dir}")


def compute_metrics(y_true, preds, probs):
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(
            precision_score(y_true, preds, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, preds, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, preds, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "far": float(compute_far(y_true, preds)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        default="data/processed/test_processed.csv"
    )
    parser.add_argument(
        "--baseline_model",
        default="results/saved_models/baseline_mlp.pt"
    )
    parser.add_argument(
        "--snn_model",
        default="results/saved_models/edge_threat_snn.pt"
    )
    parser.add_argument(
        "--output",
        default="results/tables/metrics.json"
    )

    args = parser.parse_args()
    main(args)
