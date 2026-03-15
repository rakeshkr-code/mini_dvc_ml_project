import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris


def main(out_path: str) -> None:
    df = load_iris(as_frame=True).frame
    df = df.rename(columns={"target": "label"})

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.out)