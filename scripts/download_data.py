import hashlib
import sys
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MIRROR = "https://raw.githubusercontent.com/defcom17/NSL_KDD/27bbbdf5c5cccf9970b8b6037a0f19dc09c5a0a9/"
FILES = {
    "KDDTrain+.txt": "1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95",
    "KDDTest+.txt": "fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, expected in FILES.items():
        target = RAW_DIR / name
        if target.exists() and sha256(target) == expected:
            print(f"{name}: present, checksum OK")
            continue
        print(f"{name}: downloading ...")
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(MIRROR + name, tmp)
        if sha256(tmp) != expected:
            tmp.unlink()
            print(f"{name}: CHECKSUM MISMATCH, file discarded", file=sys.stderr)
            ok = False
            continue
        tmp.replace(target)
        print(f"{name}: saved, checksum OK")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
