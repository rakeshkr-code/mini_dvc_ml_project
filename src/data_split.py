import argparse
from pathlib import Path
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

def str2bool(v):
    return str(v).lower() in ("true", "1", "yes", "y")

def load_params(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(params_path, in_path, train_out, test_out):
    params = load_params(params_path)

    seed = params["seed"]
    test_size = params["split"]["test_size"]
    stratify = params["split"]["stratify"]

    df = pd.read_csv(in_path)
    X = df.drop(columns=["label"])
    y = df["label"]

    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat
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
    parser.add_argument("--params", required=True)
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--test-out", required=True)
    args = parser.parse_args()

    main(args.params, args.in_path, args.train_out, args.test_out)