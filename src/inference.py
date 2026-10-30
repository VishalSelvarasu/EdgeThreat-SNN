import joblib
import numpy as np
import pandas as pd

import config
from data import CATEGORICAL, FEATURES
from utils import load_neural, model_path, predict_scores, to_probability


class Predictor:
    def __init__(self, kind="snn", seed=0, model_file=None, preprocessor_file=None,
                 draws=config.EVAL_SPIKE_DRAWS, threshold=0.5):
        self.model, ckpt = load_neural(model_file or model_path(kind, seed))
        self.preprocessor = joblib.load(
            preprocessor_file or config.PREPROCESSOR_FILE)
        expected = len(self.preprocessor.get_feature_names_out())
        if ckpt["input_dim"] != expected:   # the check that would have caught v1's demo crash
            raise ValueError(
                f"model expects {ckpt['input_dim']} inputs, preprocessor produces {expected}")
        self.draws = draws
        self.threshold = threshold

    @staticmethod
    def parse_line(line):
        """41 raw values, comma- or tab-separated; anything after the 41st is ignored."""
        sep = "\t" if line.count("\t") > line.count(",") else ","
        values = [v.strip() for v in line.strip().split(sep)]
        if len(values) < len(FEATURES):
            raise ValueError(
                f"expected at least {len(FEATURES)} values, got {len(values)}")
        record = dict(zip(FEATURES, values[:len(FEATURES)]))
        for name in FEATURES:
            if name not in CATEGORICAL:
                record[name] = float(record[name])
        return record

    def predict(self, records, seed=0):
        X = np.array(self.preprocessor.transform(
            pd.DataFrame(records)[FEATURES]), dtype=np.float32)
        probs = to_probability(predict_scores(
            self.model, X, draws=self.draws, seed=seed))
        return [{"prediction": "attack" if p >= self.threshold else "normal",
                 "attack_probability": round(float(p), 4)} for p in probs]

    def predict_line(self, line, seed=0):
        return self.predict([self.parse_line(line)], seed=seed)[0]
