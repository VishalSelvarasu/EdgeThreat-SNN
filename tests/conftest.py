import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import FEATURES  # noqa: E402

LABELS = ["normal", "neptune", "satan", "guess_passwd", "rootkit"]


def make_rows(n, seed, with_difficulty=True, binary_labels=False):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        label = LABELS[i % len(LABELS)]
        attack = label != "normal"
        values = []
        for f in FEATURES:
            if f == "protocol_type":
                values.append(rng.choice(["tcp", "udp", "icmp"]))
            elif f == "service":
                values.append(rng.choice(
                    ["http", "private", "ftp_data", "imap4"]))
            elif f == "flag":
                values.append("REJ" if attack else "SF")
            elif f == "src_bytes":
                values.append(
                    str(rng.randint(0, 40) if attack else rng.randint(1_000, 1_000_000)))
            elif f.endswith("rate"):
                values.append(f"{rng.random():.2f}")
            else:
                values.append(str(rng.randint(0, 5)))
        if binary_labels:
            values.append("Attack" if attack else "Normal")
        else:
            values.append(label)
        if with_difficulty:
            values.append(str(rng.randint(0, 21)))
        rows.append(values)
    return rows


def write_rows(path, rows, sep=","):
    path.write_text("\n".join(sep.join(r) for r in rows) + "\n")
    return path


@pytest.fixture
def nsl_files(tmp_path):
    train = write_rows(tmp_path / "train.txt", make_rows(500, seed=0))
    test = write_rows(tmp_path / "test.txt", make_rows(150, seed=1))
    return train, test
