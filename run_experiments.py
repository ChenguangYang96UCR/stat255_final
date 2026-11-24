#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description="Run the controlled core model comparison.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--task", choices=["binary", "multiclass"])
    parser.add_argument("--output-root", default="outputs/core")
    parser.add_argument("--quick", action="store_true", help="Run image, concat, and gated only.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    experiments = CORE_EXPERIMENTS
    if args.quick:
        experiments = [e for e in experiments if e[0] in ("image", "concat", "gated")]
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    for name, model_type, fusion, alignment in experiments:
        command = [sys.executable, "train.py", "--config", args.config, "--model-type", model_type,
                   "--alignment-weight", str(alignment), "--output-dir", str(Path(args.output_root) / name)]
        if args.task:
            command += ["--task", args.task]
        if fusion:
            command += ["--fusion", fusion]
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

