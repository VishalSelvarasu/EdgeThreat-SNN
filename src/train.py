import argparse
import copy
import csv
import time

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import config
from data import load_dataset, save_preprocessor
from models import NEURAL_KINDS, SKLEARN_KINDS, build_model, count_parameters
from utils import model_path, save_neural, set_seed

ALL_KINDS = NEURAL_KINDS + SKLEARN_KINDS


def seeds_for(kind, seeds):
    # logistic regression is deterministic
    return seeds[:1] if kind == "logreg" else seeds


def default_snn_kwargs():
    return {"time_steps": config.TIME_STEPS, "beta": config.BETA, "encoding": config.ENCODING}


@torch.no_grad()
def validation_loss(model, X_val, y_val, seed):
    model.eval()
    gen = torch.Generator().manual_seed(
        1_000_003 + seed)   # same spike draw every epoch
    total = 0.0
    for i in range(0, len(X_val), 4096):
        logits = model(X_val[i:i + 4096], generator=gen)
        total += F.cross_entropy(logits,
                                 y_val[i:i + 4096], reduction="sum").item()
    return total / len(X_val)


def train_neural(kind, seed, data, snn_kwargs=None, max_epochs=None, patience=config.PATIENCE, log=print):
    set_seed(seed)
    snn_kwargs = (snn_kwargs or default_snn_kwargs()) if kind == "snn" else {}
    model = build_model(kind, data.n_features, **snn_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    max_epochs = max_epochs or config.MAX_EPOCHS

    # fresh spikes every batch, reproducible
    spike_gen = torch.Generator().manual_seed(seed)
    order_rng = np.random.default_rng(seed)
    X, y = torch.from_numpy(data.X_train), torch.from_numpy(data.y_train)
    X_val, y_val = torch.from_numpy(data.X_val), torch.from_numpy(data.y_val)

    best_loss, best_state, best_epoch, bad = float("inf"), None, 0, 0
    start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.from_numpy(order_rng.permutation(len(X)))
        running = 0.0
        for i in range(0, len(X), config.BATCH_SIZE):
            idx = perm[i:i + config.BATCH_SIZE]
            loss = F.cross_entropy(model(X[idx], generator=spike_gen), y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(idx)
        val_loss = validation_loss(model, X_val, y_val, seed)
        log(f"  {kind} seed {seed} epoch {epoch:2d}: train {running / len(X):.4f}  val {val_loss:.4f}")
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, best_epoch, bad = val_loss, copy.deepcopy(
                model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    info = {"kind": kind, "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
            "best_val_loss": best_loss, "train_seconds": round(time.time() - start, 1),
            "parameters": count_parameters(model)}
    return model, info


def train_sklearn(kind, seed, data):
    start = time.time()
    if kind == "logreg":
        model = LogisticRegression(max_iter=2000)
    else:
        model = RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1)
    model.fit(data.X_train, data.y_train)
    info = {"kind": kind, "seed": seed, "best_epoch": None, "epochs_run": None,
            "best_val_loss": None, "train_seconds": round(time.time() - start, 1), "parameters": None}
    return model, info


def main(kinds, seeds, force=False, max_epochs=None):
    data = load_dataset(log_scale=config.LOG_SCALE)
    save_preprocessor(data)
    rows = []
    for kind in kinds:
        for seed in seeds_for(kind, seeds):
            path = model_path(kind, seed)
            if path.exists() and not force:
                print(f"skip {path.name} (exists; --force retrains)")
                continue
            print(f"training {kind} seed {seed}")
            if kind in NEURAL_KINDS:
                model, info = train_neural(
                    kind, seed, data, max_epochs=max_epochs)
                save_neural(model, kind, data.n_features, info, path)
            else:
                model, info = train_sklearn(kind, seed, data)
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, path, compress=3)
            rows.append(info)

    if rows:   # merge into the training log, replacing older rows for the same model and seed
        config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
        out = config.TABLE_DIR / "training_log.csv"
        new_keys = {(r["kind"], str(r["seed"])) for r in rows}
        old = []
        if out.exists():
            with open(out, newline="") as f:
                old = [r for r in csv.DictReader(f) if (
                    r["kind"], r["seed"]) not in new_keys]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(old + rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+",
                    default=list(ALL_KINDS), choices=ALL_KINDS)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="override, for quick tests only")
    args = ap.parse_args()
    main(args.kinds, args.seeds, args.force, args.max_epochs)
