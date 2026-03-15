import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def str2bool(v):
    return str(v).lower() in ("true", "1", "yes", "y")


def main(in_path, train_out, test_out, seed, test_size, stratify):
    df = pd.read_csv(in_path)
    X = df.drop(columns=["label"])
    y = df["label"]

    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=strat,
    )

    train_df = X_train.copy()
    train_df["label"] = y_train.values

    test_df = X_test.copy()
    test_df["label"] = y_test.values

    Path(train_out).parent.mkdir(parents=True, exist_ok=True)
    Path(test_out).parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--test-out", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--test-size", type=float, required=True)
    parser.add_argument("--stratify", type=str2bool, required=True)
    args = parser.parse_args()

    main(
        args.in_path,
        args.train_out,
        args.test_out,
        args.seed,
        args.test_size,
        args.stratify,
    )