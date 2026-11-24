#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


CORE_EXPERIMENTS = [
    ("metadata_linear", "metadata_linear", None, 0.0),
    ("metadata_mlp", "metadata_mlp", None, 0.0),
    ("image", "image", None, 0.0),
    ("concat", "multimodal", "concat", 0.0),
    ("gated", "multimodal", "gated", 0.0),
    ("decision", "multimodal", "decision", 0.0),
    ("attention", "multimodal", "attention", 0.0),
    ("concat_align", "multimodal", "concat", 0.1),
    ("gated_align", "multimodal", "gated", 0.1),
    ("attention_align", "multimodal", "attention", 0.1),
]

SUMMARY_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "auroc",
    "brier",
    "ece",
    "concerning_sensitivity",
    "concerning_specificity",
    "melanoma_sensitivity",
    "selected_epoch",
]


def aggregate_results(output_root, experiment_names, seeds):
    output_root = Path(output_root)
    rows = []
    for experiment in experiment_names:
        for seed in seeds:
            metrics_path = output_root / experiment / f"seed_{seed}" / "test_metrics.json"
            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing result file: {metrics_path}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            row = {"experiment": experiment, "seed": seed}
            for metric in SUMMARY_METRICS:
                value = metrics.get(metric)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[metric] = value
            rows.append(row)

    per_seed = pd.DataFrame(rows).sort_values(["experiment", "seed"])
    per_seed.to_csv(output_root / "per_seed_results.csv", index=False)

    summary_rows = []
    for experiment, group in per_seed.groupby("experiment", sort=False):
        summary = {"experiment": experiment, "runs": len(group)}
        for metric in SUMMARY_METRICS:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            summary[f"{metric}_mean_std"] = f"{mean:.4f} +/- {std:.4f}"
        summary_rows.append(summary)

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output_root / "summary_mean_std.csv", index=False)
    (output_root / "summary_mean_std.json").write_text(
        json.dumps(summary_rows, indent=2), encoding="utf-8"
    )

    display_columns = [
        column for column in (
            "experiment",
            "balanced_accuracy_mean_std",
            "macro_f1_mean_std",
            "auroc_mean_std",
            "concerning_sensitivity_mean_std",
            "concerning_specificity_mean_std",
        ) if column in summary_frame
    ]
    print("\nMean +/- std across seeds:")
    print(summary_frame[display_columns].to_string(index=False))
    print(f"\nSaved {output_root / 'per_seed_results.csv'}")
    print(f"Saved {output_root / 'summary_mean_std.csv'}")
    print(f"Saved {output_root / 'summary_mean_std.json'}")
    return per_seed, summary_frame


def main():
    parser = argparse.ArgumentParser(description="Run each model with several random seeds.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--task", choices=["binary", "multiclass"])
    parser.add_argument("--output-root", default="outputs/core")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 8888])
    parser.add_argument("--quick", action="store_true", help="Run image, concat, and gated only.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiments = CORE_EXPERIMENTS
    if args.quick:
        experiments = [e for e in experiments if e[0] in ("image", "concat", "gated")]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, model_type, fusion, alignment in experiments:
        for seed in args.seeds:
            output_dir = output_root / name / f"seed_{seed}"
            command = [
                sys.executable,
                "train.py",
                "--config", args.config,
                "--model-type", model_type,
                "--alignment-weight", str(alignment),
                "--seed", str(seed),
                "--output-dir", str(output_dir),
            ]
            if args.task:
                command += ["--task", args.task]
            if fusion:
                command += ["--fusion", fusion]
            print(" ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)

    if not args.dry_run:
        aggregate_results(output_root, [experiment[0] for experiment in experiments], args.seeds)


if __name__ == "__main__":
    main()
