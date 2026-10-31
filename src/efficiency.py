import json
import platform
import time

import numpy as np
import pandas as pd
import torch

import config
from data import load_dataset
from models import DISPLAY_NAMES, EdgeThreatSNN
from utils import load_any


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


@torch.no_grad()
def latency_us(model, X, n=1000, warmup=100):
    gen = torch.Generator().manual_seed(0)
    times = []
    for i in range(warmup + n):
        x = X[i % len(X):i % len(X) + 1]
        start = time.perf_counter()
        model(x, generator=gen)
        times.append(time.perf_counter() - start)
    return float(np.median(times[warmup:]) * 1e6)


@torch.no_grad()
def throughput(model, X, batch=1024, repeats=20):
    gen = torch.Generator().manual_seed(0)
    x = X[:batch]
    model(x, generator=gen)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(x, generator=gen)
        times.append(time.perf_counter() - start)
    return float(batch / np.median(times))


def linear_layers(model):
    return [m for m in model.modules() if isinstance(m, torch.nn.Linear)]


def operation_count(kind, model, runs):
    """Arithmetic operations per prediction: multiply-accumulates (MAC) and accumulates (AC)."""
    if not isinstance(model, EdgeThreatSNN):
        macs = sum(l.in_features * l.out_features for l in linear_layers(model))
        return {"mac": float(macs), "ac": 0.0}
    snn_runs = runs[runs["kind"] == kind]
    input_events = snn_runs["input_events_per_sample"].mean()
    hidden_spikes = snn_runs["hidden_spikes_per_sample"].mean()
    T, H, O = model.time_steps, model.fc1.out_features, model.fc2.out_features
    # Event-driven: every input spike adds one weight to each of the H hidden neurons,
    # every hidden spike adds one weight to each of the O outputs; biases are added once per step.
    ac = input_events * H + hidden_spikes * O + T * (H + O)
    # Each neuron's membrane decays (one multiply-add) at every time step.
    mac = T * (H + O)
    return {"mac": float(mac), "ac": float(ac)}


def main():
    torch.set_num_threads(1)
    data = load_dataset(log_scale=config.LOG_SCALE)
    X = torch.from_numpy(data.X_test)
    runs = pd.read_csv(config.TABLE_DIR / "runs.csv")
    results = {"cpu": cpu_name(), "threads": 1,
               "torch": torch.__version__, "models": {}}

    for kind in ("mlp", "mlp_small", "snn"):
        model = load_any(kind, 0)
        ops = operation_count(kind, model, runs)
        energy_nj = (ops["mac"] * config.ENERGY_PJ["mac"] +
                     ops["ac"] * config.ENERGY_PJ["ac"]) / 1000
        results["models"][kind] = {
            "latency_batch1_us": latency_us(model, X),
            "throughput_rows_per_s": throughput(model, X),
            "macs_per_prediction": ops["mac"],
            "acs_per_prediction": ops["ac"],
            "estimated_energy_nj": energy_nj,
        }
        print(kind, results["models"][kind])

    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (config.TABLE_DIR / "efficiency.json").write_text(json.dumps(results, indent=1))
    lines = [
        f"CPU: {results['cpu']}, 1 thread, PyTorch {results['torch']}. "
        "Energy uses 4.6 pJ per MAC and 0.9 pJ per AC (45 nm) and ignores memory access.\n",
        "| Model | Latency, batch 1 (µs) | Throughput (rows/s) | MACs / prediction | ACs / prediction | Estimated energy (nJ) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind, r in results["models"].items():
        lines.append(f"| {DISPLAY_NAMES[kind]} | {r['latency_batch1_us']:,.0f} | {r['throughput_rows_per_s']:,.0f} | "
                     f"{r['macs_per_prediction']:,.0f} | {r['acs_per_prediction']:,.0f} | {r['estimated_energy_nj']:.1f} |")
    (config.TABLE_DIR / "efficiency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
