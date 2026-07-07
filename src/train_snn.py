import os
import json
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from encode_spikes import rate_encode
from model_snn import EdgeThreatSNN

EPOCHS = 20
LEARNING_RATE = 1e-3
TIME_STEPS = 25


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def compute_far(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def main(args):
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    X_train = train_df.drop(columns=["label"]).values
    y_train = torch.tensor(train_df["label"].values, dtype=torch.long)

    X_test = test_df.drop(columns=["label"]).values
    y_test = torch.tensor(test_df["label"].values, dtype=torch.long)

    x_train_spikes = torch.tensor(rate_encode(X_train, TIME_STEPS), dtype=torch.float32)
    x_test_spikes = torch.tensor(rate_encode(X_test, TIME_STEPS), dtype=torch.float32)

    model = EdgeThreatSNN(input_dim=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        mem_rec = model(x_train_spikes)
        logits = mem_rec.sum(dim=0)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        mem_rec_test = model(x_test_spikes)
        latency_ms = (time.perf_counter() - start) * 1000

        logits_test = mem_rec_test.sum(dim=0)
        probs = torch.softmax(logits_test, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits_test, dim=1).cpu().numpy()

    y_true = y_test.cpu().numpy()

    avg_spike_activity = float(x_test_spikes.mean().item())

    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "far": float(compute_far(y_true, preds)),
        "latency_ms": float(latency_ms),
        "avg_spike_activity": avg_spike_activity
    }

    ensure_dir(os.path.dirname(args.output))
    ensure_dir(os.path.dirname(args.metrics_out))

    torch.save(model.state_dict(), args.output)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved SNN model to {args.output}")
    print(f"Saved SNN metrics to {args.metrics_out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train_processed.csv")
    parser.add_argument("--test", default="data/processed/test_processed.csv")
    parser.add_argument("--output", default="results/saved_models/edge_threat_snn.pt")
    parser.add_argument("--metrics_out", default="results/tables/snn_metrics.json")
    args = parser.parse_args()
    main(args)