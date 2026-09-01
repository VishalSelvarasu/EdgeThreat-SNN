import os
import json
import argparse
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from model_baseline import BaselineMLP
from config import EPOCHS, LEARNING_RATE, RANDOM_STATE
from utils import ensure_dir, set_all_seeds, compute_far


def main(args):
    set_all_seeds()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    X_train = torch.tensor(train_df.drop(
        columns=["label"]).values, dtype=torch.float32)
    y_train = torch.tensor(train_df["label"].values, dtype=torch.long)

    X_test = torch.tensor(test_df.drop(
        columns=["label"]).values, dtype=torch.float32)
    y_test = torch.tensor(test_df["label"].values, dtype=torch.long)

    model = BaselineMLP(X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(test_logits, dim=1).cpu().numpy()

    y_true = y_test.cpu().numpy()

    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "far": float(compute_far(y_true, preds))
    }

    ensure_dir(os.path.dirname(args.output))
    ensure_dir(os.path.dirname(args.metrics_out))

    torch.save(model.state_dict(), args.output)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved baseline model to {args.output}")
    print(f"Saved baseline metrics to {args.metrics_out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", default="data/processed/train_processed.csv")
    parser.add_argument("--test", default="data/processed/test_processed.csv")
    parser.add_argument(
        "--output", default="results/saved_models/baseline_mlp.pt")
    parser.add_argument(
        "--metrics_out", default="results/tables/baseline_metrics.json")
    args = parser.parse_args()
    main(args)
