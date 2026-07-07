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

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | FAR | Latency (ms) | Avg Spike Activity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline MLP | 0.7872 | 0.9590 | 0.6542 | 0.7778 | 0.9176 | 0.0370 | 2.92 | - |
| EdgeThreat-SNN | 0.7131 | 0.9580 | 0.5187 | 0.6730 | 0.8680 | 0.0301 | 181.18 | 0.0666 |

## Result summary

On the current setup, the baseline MLP performs better overall than the SNN. It reaches higher recall, F1-score, and ROC-AUC on the NSL-KDD test set.

The SNN keeps similarly high precision and slightly lowers the false alarm rate, but it misses more attacks and is much slower in the current CPU-based run. At this stage, the SNN is better treated as the experimental part of the project, while the baseline is the stronger practical model.

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