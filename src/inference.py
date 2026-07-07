import torch
import pandas as pd
from config import FEATURE_COLUMNS, TIME_STEPS
from encode_spikes import rate_encode
from model_snn import EdgeThreatSNN


def predict_snn_from_dict(sample_dict, model_path="results/saved_models/edge_threat_snn.pt"):
    row = pd.DataFrame([sample_dict])[FEATURE_COLUMNS]
    spikes = torch.tensor(rate_encode(row.values, TIME_STEPS), dtype=torch.float32)
    model = EdgeThreatSNN(input_dim=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    mem_rec = model(spikes)
    logits = mem_rec.sum(dim=0)
    probs = torch.softmax(logits, dim=1).detach().numpy()[0]
    pred = int(probs.argmax())
    return {
        "label": "attack" if pred == 1 else "normal",
        "prob_normal": float(probs[0]),
        "prob_attack": float(probs[1])
    }
