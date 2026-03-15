import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def load_params(params_path: str) -> dict:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(params_path, test_path, model_path, metrics_out, confmat_out):
    params = load_params(params_path)
    average = params["eval"]["average"]

    df = pd.read_csv(test_path)
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    bundle = joblib.load(model_path)
    scaler = bundle["scaler"]
    clf = bundle["clf"]

    Xs = scaler.transform(X)
    pred = clf.predict(Xs)

    labels = np.unique(np.concatenate([y, pred]))
    cm = confusion_matrix(y, pred, labels=labels)

    metrics = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, average=average, zero_division=0),
        "recall": recall_score(y, pred, average=average, zero_division=0),
        "f1": f1_score(y, pred, average=average, zero_division=0),
        "average": average,
    }

    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(confmat_out).parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(confmat_out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--confmat-out", required=True)
    args = parser.parse_args()

    main(args.params, args.test, args.model, args.metrics_out, args.confmat_out)