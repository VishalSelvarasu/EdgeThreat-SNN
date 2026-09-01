import os
import argparse
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from joblib import dump

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

CAT_COLS = ["protocol_type", "service", "flag"]


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def load_nsl_kdd(path):
    # Your files are tab-delimited, not comma-delimited [69][70]
    df = pd.read_csv(path, sep="\t", header=None,
                     names=NSL_KDD_COLUMNS, engine="python")
    df = df.drop(columns=["difficulty"])
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.lower()
    )
    df["label"] = (df["label"] != "normal").astype(int)
    return df


def main(args):
    train_df = load_nsl_kdd(args.train_input)
    test_df = load_nsl_kdd(args.test_input)

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    num_cols = [c for c in X_train.columns if c not in CAT_COLS]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
    train_processed_df["label"] = y_train.values

    test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)
    test_processed_df["label"] = y_test.values

    ensure_dir(os.path.dirname(args.train_output))
    ensure_dir(os.path.dirname(args.test_output))
    ensure_dir(os.path.dirname(args.preprocessor_out))

    train_processed_df.to_csv(args.train_output, index=False)
    test_processed_df.to_csv(args.test_output, index=False)
    dump(preprocessor, args.preprocessor_out)

    print(f"Saved processed train data to {args.train_output}")
    print(f"Saved processed test data to {args.test_output}")
    print(f"Saved preprocessor to {args.preprocessor_out}")
    print(f"Train shape: {train_processed_df.shape}")
    print(f"Test shape: {test_processed_df.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", default="data/raw/KDDTrain+.txt")
    parser.add_argument("--test_input", default="data/raw/KDDTest+.txt")
    parser.add_argument(
        "--train_output", default="data/processed/train_processed.csv")
    parser.add_argument(
        "--test_output", default="data/processed/test_processed.csv")
    parser.add_argument("--preprocessor_out",
                        default="results/saved_models/preprocessor.joblib")
    args = parser.parse_args()
    main(args)
