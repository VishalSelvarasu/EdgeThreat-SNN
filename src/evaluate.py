import argparse
import json
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from config import FEATURE_COLUMNS, LABEL_COLUMN, RANDOM_STATE, TIME_STEPS
from encode_spikes import rate_encode
from model_baseline import BaselineMLP
from model_snn import EdgeThreatSNN
from utils import compute_metrics, ensure_dir, now_ms


def main(args):
    df = pd.read_csv(args.data)
    X = df[FEATURE_COLUMNS].values
    y = df[LABEL_COLUMN].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

    baseline = BaselineMLP(input_dim=X.shape[1])
    baseline.load_state_dict(torch.load(args.baseline_model, map_location="cpu"))
    baseline.eval()
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    t0 = now_ms()
    baseline_logits = baseline(x_test_tensor)
    baseline_latency = now_ms() - t0
    baseline_probs = torch.softmax(baseline_logits, dim=1)[:, 1].detach().numpy()
    baseline_preds = baseline_logits.argmax(dim=1).detach().numpy()

    snn = EdgeThreatSNN(input_dim=X.shape[1])
    snn.load_state_dict(torch.load(args.snn_model, map_location="cpu"))
    snn.eval()
    x_test_spikes = torch.tensor(rate_encode(X_test, TIME_STEPS), dtype=torch.float32)
    t1 = now_ms()
    mem_rec = snn(x_test_spikes)
    snn_latency = now_ms() - t1
    snn_logits = mem_rec.sum(dim=0)
    snn_probs = torch.softmax(snn_logits, dim=1)[:, 1].detach().numpy()
    snn_preds = snn_logits.argmax(dim=1).detach().numpy()

    results = {
        "baseline_mlp": compute_metrics(y_test, baseline_preds, baseline_probs),
        "edge_threat_snn": compute_metrics(y_test, snn_preds, snn_probs)
    }
    results["baseline_mlp"]["latency_ms"] = baseline_latency
    results["edge_threat_snn"]["latency_ms"] = snn_latency
    results["edge_threat_snn"]["avg_spike_activity"] = float((x_test_spikes.numpy() > 0).mean())
    ensure_dir("results/tables")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--baseline_model", required=True)
    parser.add_argument("--snn_model", required=True)
    parser.add_argument("--output", default="results/tables/metrics.json")
    main(parser.parse_args())
