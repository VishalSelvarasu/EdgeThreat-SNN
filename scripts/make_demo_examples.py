import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = [l.rstrip("\n").split(",") for l in open(
    ROOT / "data" / "raw" / "KDDTest+.txt", encoding="utf-8")]
picked = [next(r for r in rows if r[41] == name) for name in
          ["normal", "neptune", "satan", "guess_passwd", "buffer_overflow", "mscan"]]
with open(ROOT / "data" / "sample" / "demo_records.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["record", "true_label"])
    for r in picked:
        w.writerow([",".join(r[:41]), r[41]])
print("wrote data/sample/demo_records.csv")
