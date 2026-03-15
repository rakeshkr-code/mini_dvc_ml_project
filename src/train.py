import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler


def main(train_path, model_out, history_out, seed, epochs, lr, alpha):
    df = pd.read_csv(train_path)
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    classes = np.unique(y)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = SGDClassifier(
        loss="log_loss",
        alpha=alpha,
        learning_rate="constant",
        eta0=lr,
        random_state=seed,
        tol=None,
    )

    rng = np.random.default_rng(seed)
    history = []

    for epoch in range(1, epochs + 1):
        idx = rng.permutation(len(Xs))
        X_epoch = Xs[idx]
        y_epoch = y[idx]

        if epoch == 1:
            clf.partial_fit(X_epoch, y_epoch, classes=classes)
        else:
            clf.partial_fit(X_epoch, y_epoch)

        proba = clf.predict_proba(Xs)
        pred = clf.predict(Xs)

        history.append(
            {
                "epoch": epoch,
                "loss": log_loss(y, proba, labels=classes),
                "accuracy": accuracy_score(y, pred),
            }
        )

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(history_out).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({"scaler": scaler, "clf": clf, "classes": classes}, model_out)
    pd.DataFrame(history).to_csv(history_out, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--history-out", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()

    main(
        args.train,
        args.model_out,
        args.history_out,
        args.seed,
        args.epochs,
        args.lr,
        args.alpha,
    )