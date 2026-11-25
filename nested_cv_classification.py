#!/usr/bin/env python3
"""Nested grouped cross-validation for all classification architectures."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nested_cv_importance import (fit_outer_model, grouped_folds, inner_validation_score,
                                  performance_summary)
from prepare_data import DEFAULT_REQUIRED_METADATA, filter_complete_samples
from src.isic_multimodal import task_classes
from src.isic_multimodal.data import load_isic_tables
from src.isic_multimodal.utils import device_from_environment, load_config


CLASSIFICATION_EXPERIMENTS = {
    "metadata_linear": {"model_type": "metadata_linear", "fusion": "concat", "alignment_weight": 0.0},
    "metadata_mlp": {"model_type": "metadata_mlp", "fusion": "concat", "alignment_weight": 0.0},
    "image": {"model_type": "image", "fusion": "concat", "alignment_weight": 0.0},
    "concat": {"model_type": "multimodal", "fusion": "concat", "alignment_weight": 0.0},
    "gated": {"model_type": "multimodal", "fusion": "gated", "alignment_weight": 0.0},
    "decision": {"model_type": "multimodal", "fusion": "decision", "alignment_weight": 0.0},
    "attention": {"model_type": "multimodal", "fusion": "attention", "alignment_weight": 0.0},
    "concat_align": {"model_type": "multimodal", "fusion": "concat", "alignment_weight": 0.1},
    "gated_align": {"model_type": "multimodal", "fusion": "gated", "alignment_weight": 0.1},
    "attention_align": {"model_type": "multimodal", "fusion": "attention", "alignment_weight": 0.1},
}


def classification_config(base_cfg, experiment, candidate):
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["metadata_features"] = ["age_approx", "sex", "anatom_site_general"]
    cfg["model"]["type"] = experiment["model_type"]
    cfg["model"]["fusion"] = experiment["fusion"]
    cfg["model"]["dropout"] = candidate["dropout"]
    cfg["training"]["learning_rate"] = candidate["learning_rate"]
    cfg["training"]["weight_decay"] = candidate["weight_decay"]
    cfg["training"]["alignment_weight"] = experiment["alignment_weight"]
    return cfg


def main():
    parser = argparse.ArgumentParser(description="5x5 nested CV classification comparison.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs/nested_cv_classification")
    parser.add_argument("--outer-folds", type=int)
    parser.add_argument("--inner-folds", type=int)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(CLASSIFICATION_EXPERIMENTS),
        default=list(CLASSIFICATION_EXPERIMENTS),
    )
    parser.add_argument("--epochs", type=int, help="Override maximum epochs for development runs.")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task = cfg["data"].get("task", "binary")
    if task != "binary":
        raise ValueError("The current nested classification study expects data.task: binary.")
    nested = cfg.get("nested_cv", {})
    outer_folds = args.outer_folds or nested.get("outer_folds", 5)
    inner_folds = args.inner_folds or nested.get("inner_folds", 5)
    candidates = nested.get("candidates")
    if not candidates:
        raise ValueError("config.yaml must define nested_cv.candidates.")
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.no_pretrained:
        cfg["model"]["pretrained"] = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = task_classes(task)
    frame = load_isic_tables(
        cfg["data"]["metadata_csv"],
        cfg["data"]["ground_truth_csv"],
        drop_unmatched=True,
        task=task,
    )
    frame, removed_metadata, removed_images = filter_complete_samples(
        frame, DEFAULT_REQUIRED_METADATA, cfg["data"]["image_dir"]
    )
    print(f"Using {len(frame)} samples; removed metadata={removed_metadata}, images={removed_images}.")

    device = device_from_environment()
    print(f"Using device: {device}")
    outer_splits = grouped_folds(frame, outer_folds, cfg["seed"])
    outer_assignment = np.full(len(frame), -1, dtype=int)
    for outer_index, (_, outer_test_idx) in enumerate(outer_splits, start=1):
        outer_assignment[outer_test_idx] = outer_index
    if (outer_assignment < 0).any():
        raise RuntimeError("Some samples were not assigned to an outer test fold.")
    assignments = frame[["image", "diagnosis", "label", "lesion_id"]].copy()
    assignments["outer_fold"] = outer_assignment
    assignments.to_csv(output_dir / "outer_fold_assignments.csv", index=False)

    inner_rows = []
    selected_rows = []
    outer_rows = []
    for experiment_name in args.experiments:
        experiment = CLASSIFICATION_EXPERIMENTS[experiment_name]
        print(f"\n===== Classifier: {experiment_name} =====")
        for outer_index, (outer_train_idx, outer_test_idx) in enumerate(outer_splits, start=1):
            outer_train = frame.iloc[outer_train_idx].reset_index(drop=True)
            outer_test = frame.iloc[outer_test_idx].reset_index(drop=True)
            inner_splits = grouped_folds(
                outer_train, inner_folds, cfg["seed"] + 1000 * outer_index
            )
            candidate_results = []

            for candidate_index, candidate in enumerate(candidates):
                fold_scores = []
                fold_epochs = []
                candidate_cfg = classification_config(cfg, experiment, candidate)
                for inner_index, (inner_train_idx, inner_val_idx) in enumerate(inner_splits, start=1):
                    inner_train = outer_train.iloc[inner_train_idx].reset_index(drop=True)
                    inner_val = outer_train.iloc[inner_val_idx].reset_index(drop=True)
                    run_seed = cfg["seed"] + outer_index * 1000 + inner_index
                    score, best_epoch = inner_validation_score(
                        candidate_cfg, inner_train, inner_val, class_names, run_seed, device
                    )
                    fold_scores.append(score)
                    fold_epochs.append(best_epoch)
                    inner_rows.append({
                        "experiment": experiment_name,
                        "outer_fold": outer_index,
                        "candidate": candidate_index,
                        "inner_fold": inner_index,
                        "balanced_accuracy": score,
                        "best_epoch": best_epoch,
                        **candidate,
                    })
                    print(
                        f"outer={outer_index} candidate={candidate_index} inner={inner_index} "
                        f"balanced_acc={score:.4f} epoch={best_epoch}"
                    )
                candidate_results.append({
                    "candidate": candidate_index,
                    "mean_score": float(np.mean(fold_scores)),
                    "mean_best_epoch": float(np.mean(fold_epochs)),
                    "config": candidate,
                })

            selected = max(candidate_results, key=lambda result: result["mean_score"])
            selected_epochs = max(1, int(round(selected["mean_best_epoch"])))
            selected_cfg = classification_config(cfg, experiment, selected["config"])
            outer_seed = cfg["seed"] + outer_index * 1000 + 999
            metrics = fit_outer_model(
                selected_cfg,
                outer_train,
                outer_test,
                class_names,
                selected_epochs,
                outer_seed,
                device,
            )
            selected_rows.append({
                "experiment": experiment_name,
                "outer_fold": outer_index,
                "selected_candidate": selected["candidate"],
                "inner_balanced_accuracy": selected["mean_score"],
                "refit_epochs": selected_epochs,
                **selected["config"],
            })
            outer_rows.append({
                "experiment": experiment_name,
                "outer_fold": outer_index,
                "n_train": len(outer_train),
                "n_test": len(outer_test),
                "selected_candidate": selected["candidate"],
                "refit_epochs": selected_epochs,
                **{key: value for key, value in metrics.items() if isinstance(value, (int, float))},
            })
            print(
                f"OUTER {outer_index}: balanced_acc={metrics['balanced_accuracy']:.4f}, "
                f"macro_f1={metrics['macro_f1']:.4f}, auroc={metrics['auroc']:.4f}"
            )

            pd.DataFrame(inner_rows).to_csv(output_dir / "inner_cv_results.csv", index=False)
            pd.DataFrame(selected_rows).to_csv(output_dir / "selected_hyperparameters.csv", index=False)
            pd.DataFrame(outer_rows).to_csv(output_dir / "outer_fold_results.csv", index=False)

    outer_results = pd.DataFrame(outer_rows)
    summary = performance_summary(outer_results)
    summary.to_csv(output_dir / "classification_summary.csv", index=False)
    (output_dir / "classification_summary.json").write_text(
        summary.to_json(orient="records", indent=2), encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(json.dumps({
        "task": task,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "experiments": args.experiments,
        "candidates": candidates,
        "fixed_cohort_size": len(frame),
        "selection_metric": "inner-fold balanced accuracy",
    }, indent=2), encoding="utf-8")

    print("\nClassification performance mean +/- std across outer folds:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
