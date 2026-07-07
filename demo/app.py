import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr
from inference import predict_snn_from_dict
from config import FEATURE_COLUMNS


def run_inference(*values):
    sample = dict(zip(FEATURE_COLUMNS, values))
    result = predict_snn_from_dict(sample)
    result["response"] = "Trigger alert and isolate node" if result["label"] == "attack" else "Continue monitoring"
    return result


demo = gr.Interface(
    fn=run_inference,
    inputs=[gr.Number(label=col, value=0) for col in FEATURE_COLUMNS],
    outputs=gr.JSON(label="EdgeThreat-SNN Output"),
    title="EdgeThreat-SNN Demo",
    description="Practical neuromorphic edge threat scoring demo."
)

if __name__ == "__main__":
    demo.launch()
