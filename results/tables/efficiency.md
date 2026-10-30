CPU: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD, 1 thread, PyTorch 2.15.0+cpu. Energy uses 4.6 pJ per MAC and 0.9 pJ per AC (45 nm) and ignores memory access.

| Model | Latency, batch 1 (µs) | Throughput (rows/s) | MACs / prediction | ACs / prediction | Estimated energy (nJ) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP (64-32) | 35 | 4,255,142 | 9,920 | 0 | 45.6 |
| MLP (64), size-matched | 24 | 5,329,170 | 7,936 | 0 | 36.5 |
| EdgeThreat-SNN | 7,114 | 28,616 | 1,650 | 17,220 | 23.1 |
