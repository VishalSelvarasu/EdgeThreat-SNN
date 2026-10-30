"""NSL-KDD loading, train/validation split and preprocessing."""
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder

import config

FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

# 'worm' counted as R2L and 'httptunnel' as U2R: this reproduces the most-cited
# KDDTest+ counts (DoS 7,458, Probe 2,421, R2L 2,754, U2R 200).
FAMILIES = {
    "DoS": ["back", "land", "neptune", "pod", "smurf", "teardrop",
            "apache2", "mailbomb", "processtable", "udpstorm"],
    "Probe": ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"],
    "R2L": ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
            "warezclient", "warezmaster", "named", "sendmail", "snmpgetattack",
            "snmpguess", "xlock", "xsnoop", "worm"],
    "U2R": ["buffer_overflow", "loadmodule", "perl", "rootkit", "ps",
            "sqlattack", "xterm", "httptunnel"],
}
ATTACK_FAMILY = {name: fam for fam, names in FAMILIES.items()
                 for name in names}


def family_of(label):
    return "normal" if label == "normal" else ATTACK_FAMILY.get(label, "unknown")


def read_nsl_kdd(path):
    """Read an NSL-KDD file: comma- or tab-separated, with (43 fields) or without (42) difficulty."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/download_data.py")
    with open(path, encoding="utf-8") as f:
        first = f.readline()
    sep = "\t" if first.count("\t") > first.count(",") else ","
    df = pd.read_csv(path, sep=sep, header=None, skipinitialspace=True)
    if df.shape[1] == 43:
        df.columns = FEATURES + ["label", "difficulty"]
    elif df.shape[1] == 42:
        df.columns = FEATURES + ["label"]
    else:
        raise ValueError(
            f"{path.name}: expected 42 or 43 fields per row, found {df.shape[1]}")

    for c in CATEGORICAL:
        df[c] = df[c].astype(str).str.strip()
    label = df["label"].astype(str).str.strip().str.rstrip(".").str.lower()
    df["attack_type"] = label
    df["family"] = label.map(family_of)
    df["y"] = (label != "normal").astype(np.int64)
    if set(label.unique()) <= {"normal", "attack"}:
        warnings.warn(f"{path.name} has only normal/attack labels; per-family results will be skipped.",
                      stacklevel=2)
    return df


def build_preprocessor(train_df, log_scale=True, categories=None):
    """Scalers are fitted on training rows only. The one-hot vocabulary comes from all of
    KDDTrain+ so a rare service that falls entirely into the validation split keeps its column."""
    log_cols = [c for c in NUMERIC if train_df[c].max() >
                1] if log_scale else []
    lin_cols = [c for c in NUMERIC if c not in log_cols]
    transformers = []
    if log_cols:
        transformers.append(("log", Pipeline([
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", MinMaxScaler(clip=True)),
        ]), log_cols))
    transformers.append(("num", MinMaxScaler(clip=True), lin_cols))
    transformers.append(("cat", OneHotEncoder(categories=categories or "auto",
                                              handle_unknown="ignore", sparse_output=False), CATEGORICAL))
    return ColumnTransformer(transformers).fit(train_df[FEATURES])


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_family: np.ndarray
    test_novel: np.ndarray        # attacks whose type never occurs in KDDTrain+
    preprocessor: ColumnTransformer = field(repr=False)

    @property
    def n_features(self):
        return self.X_train.shape[1]


def load_dataset(log_scale=True, train_file=None, test_file=None):
    train_full = read_nsl_kdd(train_file or config.TRAIN_FILE)
    test = read_nsl_kdd(test_file or config.TEST_FILE)
    train, val = train_test_split(train_full, test_size=config.VAL_FRACTION,
                                  stratify=train_full["y"], random_state=config.SPLIT_SEED)
    vocabulary = [sorted(train_full[c].unique()) for c in CATEGORICAL]
    pre = build_preprocessor(train, log_scale=log_scale, categories=vocabulary)

    def X(df):  # np.array(...) gives a writable copy (pandas 3 returns read-only arrays)
        return np.array(pre.transform(df[FEATURES]), dtype=np.float32)

    def col(df, name):
        return np.array(df[name].to_numpy())

    known = set(train_full["attack_type"])
    return Dataset(
        X_train=X(train), y_train=col(train, "y"),
        X_val=X(val), y_val=col(val, "y"),
        X_test=X(test), y_test=col(test, "y"),
        test_family=col(test, "family"),
        test_novel=(col(test, "y") == 1) & ~np.array(
            test["attack_type"].isin(known).to_numpy()),
        preprocessor=pre,
    )


def save_preprocessor(data):
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(data.preprocessor, config.PREPROCESSOR_FILE)


if __name__ == "__main__":
    d = load_dataset()
    print(
        f"features {d.n_features} | train {len(d.y_train)} val {len(d.y_val)} test {len(d.y_test)}")
    print(
        f"attack share: train {d.y_train.mean():.3f} val {d.y_val.mean():.3f} test {d.y_test.mean():.3f}")
    print("test families:", {f: int((d.test_family == f).sum())
          for f in ["normal", "DoS", "Probe", "R2L", "U2R", "unknown"]})
    print(f"test attacks of unseen types: {int(d.test_novel.sum())}")
    print(
        f"value range: {d.X_test.min():.3f} to {d.X_test.max():.3f}, mean {d.X_test.mean():.4f}")
