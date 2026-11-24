#!/usr/bin/env python3
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score


def score(y, pred, metric):
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y, pred)
    return f1_score(y, pred, average="macro", zero_division=0)


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap comparison of two test prediction files.")
    parser.add_argument("model_a")
    parser.add_argument("model_b")
    parser.add_argument("--metric", choices=["balanced_accuracy", "macro_f1"], default="balanced_accuracy")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    a = pd.read_csv(args.model_a)
    b = pd.read_csv(args.model_b)
    merged = a[["image", "label", "prediction"]].merge(
        b[["image", "label", "prediction"]], on="image", suffixes=("_a", "_b"), validate="one_to_one")
    if not (merged["label_a"] == merged["label_b"]).all():
        raise ValueError("The two files disagree on true labels.")
    y = merged["label_a"].to_numpy()
    pa, pb = merged["prediction_a"].to_numpy(), merged["prediction_b"].to_numpy()
    observed = score(y, pb, args.metric) - score(y, pa, args.metric)
    rng = np.random.default_rng(args.seed)
    differences = []
    for _ in range(args.samples):
        idx = rng.integers(0, len(y), len(y))
        differences.append(score(y[idx], pb[idx], args.metric) - score(y[idx], pa[idx], args.metric))
    lower, upper = np.percentile(differences, [2.5, 97.5])
    result = {"metric": args.metric, "difference_b_minus_a": float(observed),
              "bootstrap_95_ci": [float(lower), float(upper)], "n_test": len(y),
              "bootstrap_samples": args.samples}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

