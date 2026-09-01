# EdgeThreat-SNN

EdgeThreat-SNN is an intrusion detection project that compares a baseline multilayer perceptron (MLP) with a spiking neural network (SNN) on the NSL-KDD dataset. The project is built around one complete workflow: preprocess the dataset, train both models, evaluate them on the same test split, and compare the results.

## Overview

The main focus of this project is to test whether a simple event-driven spiking model can be used in a practical intrusion detection pipeline. Instead of treating the SNN as a standalone idea, the repository compares it directly against a standard neural baseline under the same preprocessing and evaluation setup.

The task is binary intrusion detection:
- `normal` -> 0
- `attack` -> 1

## Dataset

This project uses the **NSL-KDD** dataset.

Files used:
- `KDDTrain+.txt`
- `KDDTest+.txt`

During preprocessing:
- explicit NSL-KDD column names are assigned,
- the `difficulty` column is removed,
- categorical features are one-hot encoded,
- numerical features are scaled,
- processed train and test CSV files are saved.

## Setup

Create a virtual environment and install the required packages.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the project

Preprocess the dataset:

```powershell
python src/preprocess.py
```

Train the baseline model:

```powershell
python src/train_baseline.py
```

Train the spiking model:

```powershell
python src/train_snn.py
```

Evaluate both models:

```powershell
python src/evaluate.py
```

Generate the comparison table:

```powershell
python src/comparison_table.py
```

The main outputs are written to:
- `data/processed/`
- `results/saved_models/`
- `results/tables/`

## Results

Current results on the processed NSL-KDD test set:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | FAR | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline MLP | 0.750 | 0.975 | 0.576 | 0.724 | 0.922 | 0.019 | 3.9 |
| EdgeThreat-SNN | 0.800 | 0.918 | 0.713 | 0.802 | 0.871 | 0.084 | 214.6 |

## Result summary

The SNN achieves higher accuracy (80.0% vs 75.0%), better recall (71.3% vs 57.6%), and a stronger F1-score (0.802 vs 0.724), meaning it catches more attacks overall. The baseline MLP maintains superior precision (97.5% vs 91.8%) and a much lower false alarm rate (1.9% vs 8.4%), which matters for operational deployment where alert fatigue is a real concern. The baseline is also substantially faster at inference time (3.9ms vs 214.6ms).

This trade-off is the key takeaway: the SNN shows better detection coverage, while the baseline is more conservative and efficient. For a production IDS, the choice between them would depend on whether catching more attacks or minimizing false alarms is the higher priority.

## Limitations

- The current SNN is a small prototype and has not been tuned heavily.
- The task is binary intrusion detection, not full attack-category classification.
- Latency was measured on a local CPU setup.
- The project uses structured intrusion data, not true event-camera surveillance data.

## Future work

- tune the SNN architecture and number of time steps,
- try other spike-encoding methods,
- extend the task to multiclass attack detection,
- add confusion matrices and ROC plots,
- test on additional intrusion-detection datasets.

## Note

This repository contains the full working pipeline used for the current experiments: preprocessing, model training, evaluation, and comparison.

## License

This project is released under the MIT License.