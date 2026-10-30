import joblib
import numpy as np
import pytest
import torch

from conftest import make_rows, write_rows
from data import FAMILIES, NUMERIC, load_dataset, read_nsl_kdd
from inference import Predictor
from models import EdgeThreatSNN, build_model, count_parameters, encode
from train import train_neural
from utils import binary_metrics, recall_at_far, save_neural, threshold_for_far


def test_loader_reads_comma_43_fields(nsl_files):
    df = read_nsl_kdd(nsl_files[0])
    assert len(df) == 500 and df["y"].mean() == pytest.approx(0.8)
    assert set(df.loc[df["attack_type"] == "neptune", "family"]) == {"DoS"}


def test_loader_reads_tab_42_fields_and_warns(tmp_path):
    path = write_rows(tmp_path / "t.txt", make_rows(50, 3,
                      with_difficulty=False, binary_labels=True), sep="\t")
    with pytest.warns(UserWarning):
        df = read_nsl_kdd(path)
    assert df["y"].mean() == pytest.approx(0.8)


def test_loader_rejects_wrong_field_count(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text(",".join(["0"] * 30) + "\n")
    with pytest.raises(ValueError):
        read_nsl_kdd(path)


def test_family_map_has_39_unique_attacks():
    names = [n for group in FAMILIES.values() for n in group]
    assert len(names) == len(set(names)) == 39


def test_features_in_unit_range(nsl_files):
    d = load_dataset(train_file=nsl_files[0], test_file=nsl_files[1])
    # protocols, services, flags in the fake data
    assert d.n_features == len(NUMERIC) + 3 + 4 + 2
    for X in (d.X_train, d.X_val, d.X_test):
        assert X.min() >= 0.0 and X.max() <= 1.0


def test_parameter_counts():
    assert count_parameters(build_model("mlp", 122)) == 10_018
    assert count_parameters(build_model("mlp_small", 122)) == 8_002
    assert count_parameters(build_model("snn", 122)) == 8_002


def test_rate_encoding_binary_seeded_unbiased():
    x = torch.full((1000, 4), 0.3)
    a = encode(x, 25, "rate", torch.Generator().manual_seed(1))
    b = encode(x, 25, "rate", torch.Generator().manual_seed(1))
    assert torch.equal(a, b) and set(a.unique().tolist()) <= {0.0, 1.0}
    assert a.mean().item() == pytest.approx(0.3, abs=0.01)


def test_recall_at_far_and_threshold():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.95])
    assert recall_at_far(y, s, 0.0) == pytest.approx(0.5)
    assert recall_at_far(y, s, 0.25) == pytest.approx(1.0)
    assert binary_metrics(y, s, threshold_for_far(y, s, 0.25))[
        "far"] == pytest.approx(0.25)


def test_predictor_end_to_end_and_mismatch(tmp_path, nsl_files):
    data = load_dataset(train_file=nsl_files[0], test_file=nsl_files[1])
    model, info = train_neural("snn", 0, data, snn_kwargs={"time_steps": 5, "beta": 0.9, "encoding": "rate"},
                               max_epochs=2, log=lambda m: None)
    save_neural(model, "snn", data.n_features, info, tmp_path / "snn.pt")
    joblib.dump(data.preprocessor, tmp_path / "pre.joblib")
    p = Predictor(model_file=tmp_path / "snn.pt",
                  preprocessor_file=tmp_path / "pre.joblib")
    line = ",".join(make_rows(1, seed=7)[0])
    out = p.predict_line(line)
    assert 0.0 <= out["attack_probability"] <= 1.0 and p.predict_line(
        line) == out

    save_neural(EdgeThreatSNN(13, time_steps=5), "snn", 13,
                {}, tmp_path / "wrong.pt")   # v1's demo bug
    with pytest.raises(ValueError):
        Predictor(model_file=tmp_path / "wrong.pt",
                  preprocessor_file=tmp_path / "pre.joblib")
