# Methodology

1. Collect NSL-KDD style network intrusion data.
2. Normalize features and map labels to binary classes.
3. Convert normalized values to spike trains using rate coding.
4. Train a feedforward LIF-based SNN in snnTorch.
5. Train a baseline MLP for comparison.
6. Evaluate both models with classification and latency metrics.
7. Demonstrate inference through a lightweight dashboard.
