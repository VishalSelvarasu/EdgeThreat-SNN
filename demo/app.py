import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402

from inference import Predictor  # noqa: E402

predictor = Predictor(kind="snn", seed=0)


def load_examples():
    with open(ROOT / "data" / "sample" / "demo_records.csv", newline="") as f:
        return [[row["record"]] for row in csv.DictReader(f)]


def score(record):
    try:
        result = predictor.predict_line(record)
    except ValueError as err:
        return {"error": str(err)}
    result["suggested_action"] = "Flag for analyst review" if result["prediction"] == "attack" else "No action"
    result["model"] = "EdgeThreat-SNN, seed 0 (averaged over 5 spike encodings)"
    return result


demo = gr.Interface(
    fn=score,
    inputs=gr.Textbox(
        label="NSL-KDD record (41 comma-separated feature values)", lines=3),
    outputs=gr.JSON(label="Result"),
    examples=load_examples(),
    flagging_mode="never",
    title="EdgeThreat-SNN",
    description="Binary intrusion scoring with a spiking neural network trained on NSL-KDD. "
                "Research prototype, not a production intrusion-detection system.",
)

if __name__ == "__main__":
    demo.launch()
