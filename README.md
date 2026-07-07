# EdgeThreat-SNN

EdgeThreat-SNN is a practical neuromorphic cyber-physical threat detection project for real-time edge surveillance. It combines spike encoding, a Spiking Neural Network (SNN), a baseline MLP, and a lightweight dashboard-style demo for practical proof-of-concept.

## Why this project matters
Modern edge systems such as smart cameras, IoT gateways, industrial CPS nodes, and autonomous surveillance endpoints need low-latency anomaly detection. Event-driven processing and SNNs are promising because they operate on sparse temporal signals and fit resource-constrained edge settings.

## Project scope
This repository implements:
- Data preprocessing for NSL-KDD style intrusion data
- Temporal spike encoding from structured features
- A baseline MLP classifier
- A feedforward LIF-based SNN using snnTorch
- Evaluation with accuracy, precision, recall, F1, ROC-AUC, FAR, and latency
- A practical Gradio demo for threat scoring

## Architecture
1. Raw cyber / cyber-physical data input
2. Cleaning and feature engineering
3. Temporal windowing and spike encoding
4. SNN inference at edge node
5. Threat score generation
6. Response recommendation and logging

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/preprocess.py --input data/sample/sample_nsl_kdd.csv --output data/sample/processed_sample.csv
python src/train_baseline.py --data data/sample/processed_sample.csv
python src/train_snn.py --data data/sample/processed_sample.csv
python src/evaluate.py --data data/sample/processed_sample.csv --baseline_model results/saved_models/baseline_mlp.pt --snn_model results/saved_models/edge_threat_snn.pt
python demo/app.py
```
