import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import FEATURE_COLUMNS, LABEL_COLUMN
from utils import label_to_binary, save_artifact, ensure_dir


def main(args):
    df = pd.read_csv(args.input)
    df = df[FEATURE_COLUMNS + [LABEL_COLUMN]].copy()
    df[LABEL_COLUMN] = label_to_binary(df[LABEL_COLUMN])
    scaler = MinMaxScaler()
    df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])
    ensure_dir(args.output.rsplit('/', 1)[0])
    df.to_csv(args.output, index=False)
    save_artifact(scaler, args.scaler_out)
    print(f"Saved processed data to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scaler_out", default="results/saved_models/scaler.joblib")
    main(parser.parse_args())
