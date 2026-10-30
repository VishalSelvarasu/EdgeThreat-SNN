import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*args):
    print(f"\n=== {' '.join(args)} ===", flush=True)
    start = time.time()
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)
    print(f"--- finished in {time.time() - start:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()

    run("scripts/download_data.py")
    run("src/train.py")
    run("src/evaluate.py")
    run("scripts/significance.py")
    run("src/efficiency.py")
    if not args.skip_ablation:
        run("src/ablation.py")
        run("src/ablation.py", "--report")
    run("src/plot_results.py")
    print("\nDone. Tables: results/tables/  Figures: results/figures/")


if __name__ == "__main__":
    main()
