import argparse
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from model_baseline import BaselineMLP
from config import FEATURE_COLUMNS, LABEL_COLUMN, RANDOM_STATE, EPOCHS, LEARNING_RATE
from utils import ensure_dir


def main(args):
    df = pd.read_csv(args.data)
    X = torch.tensor(df[FEATURE_COLUMNS].values, dtype=torch.float32)
    y = torch.tensor(df[LABEL_COLUMN].values, dtype=torch.long)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    model = BaselineMLP(X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
    ensure_dir("results/saved_models")
    torch.save(model.state_dict(), args.output)
    print(f"Saved baseline model to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results/saved_models/baseline_mlp.pt")
    main(parser.parse_args())
