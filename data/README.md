# Data

The NSL-KDD files are not stored in this repository. Download them with:

```
python scripts/download_data.py
```

The script fetches `KDDTrain+.txt` and `KDDTest+.txt` from a public mirror pinned to one commit (`defcom17/NSL_KDD@27bbbdf5c5cccf9970b8b6037a0f19dc09c5a0a9`) and checks each file against the SHA-256 hash of the original release. A download that does not match is discarded.

| File | Rows | SHA-256 |
| --- | ---: | --- |
| KDDTrain+.txt | 125,973 | `1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95` |
| KDDTest+.txt | 22,544 | `fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84` |

Each row has 43 comma-separated fields: 41 features, the attack label and a difficulty score. Source: Canadian Institute for Cybersecurity, University of New Brunswick, <https://www.unb.ca/cic/datasets/nsl.html>. The dataset has its own terms of use and is not covered by this repository's license.

`sample/demo_records.csv` holds the six example records used by the demo, written by `scripts/make_demo_examples.py`.