#!/usr/bin/env python3
"""Nested grouped cross-validation for input ablation importance."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from prepare_data import DEFAULT_REQUIRED_METADATA, filter_complete_samples
from src.isic_multimodal import task_classes
from src.isic_multimodal.data import (ISICDataset, MetadataPreprocessor, image_transforms,
                                      load_isic_tables)
from src.isic_multimodal.engine import predict, run_epoch
from src.isic_multimodal.losses import build_loss, class_weights
from src.isic_multimodal.metrics import calculate_metrics
from src.isic_multimodal.utils import device_from_environment, load_config, seed_everything


ABLATIONS = {
    "full": {"include_image": True, "metadata_features": ["age_approx", "sex", "anatom_site_general"]},
    "no_image": {"include_image": False, "metadata_features": ["age_approx", "sex", "anatom_site_general"]},
    "no_age": {"include_image": True, "metadata_features": ["sex", "anatom_site_general"]},
    "no_sex": {"include_image": True, "metadata_features": ["age_approx", "anatom_site_general"]},
    "no_site": {"include_image": True, "metadata_features": ["age_approx", "sex"]},
}

INPUT_FROM_ABLATION = {
    "no_image": "image",
    "no_age": "age_approx",
    "no_sex": "sex",
    "no_site": "anatom_site_general",
}

IMPORTANCE_METRICS = ["balanced_accuracy", "macro_f1", "auroc"]


def grouped_folds(frame, n_splits, seed):
    groups = frame["lesion_id"].astype(str).to_numpy()
    labels = frame["label"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(frame, labels, groups))


def make_loader(cfg, frame, preprocessor, train, require_images, seed):
    metadata = preprocessor.transform(frame)
    dataset = ISICDataset(
        frame,
        cfg["data"]["image_dir"],
        metadata,
        image_transforms(cfg["data"]["image_size"], train=train),
        require_images=require_images,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=train,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg["data"]["num_workers"] > 0,
        generator=generator,
    )


def experiment_config(base_cfg, ablation, candidate):
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["metadata_features"] = list(ablation["metadata_features"])
    cfg["model"]["type"] = "multimodal" if ablation["include_image"] else "metadata_mlp"
    cfg["model"]["fusion"] = "concat"
    cfg["model"]["dropout"] = candidate["dropout"]
    cfg["training"]["learning_rate"] = candidate["learning_rate"]
    cfg["training"]["weight_decay"] = candidate["weight_decay"]
    cfg["training"]["alignment_weight"] = 0.0
    return cfg


def build_training_objects(cfg, train_frame, preprocessor, class_names, device):
    from src.isic_multimodal.models import build_model

    model = build_model(cfg, preprocessor.output_dim, len(class_names)).to(device)
    weights = class_weights(train_frame["label"].to_numpy(), len(class_names)).to(device)
    use_weights = cfg["training"]["loss"] in ("weighted_ce", "focal")
    criterion = build_loss(
        cfg["training"]["loss"],
        weights if use_weights else None,
        cfg["training"]["focal_gamma"],
    )

    backbone_ids = set()
    if hasattr(model, "image_encoder"):
        backbone_ids = {id(parameter) for parameter in model.image_encoder.features.parameters()}
    head_parameters = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    backbone_parameters = [parameter for parameter in model.parameters() if id(parameter) in backbone_ids]
    parameter_groups = [{"params": head_parameters, "lr": cfg["training"]["learning_rate"]}]
    if backbone_parameters:
        parameter_groups.append({
            "params": backbone_parameters,
            "lr": cfg["training"]["backbone_learning_rate"],
        })
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=cfg["training"]["weight_decay"])
    return model, criterion, optimizer


def set_backbone_state(model, frozen):
    if hasattr(model, "image_encoder"):
        model.image_encoder.freeze_backbone(frozen)


def inner_validation_score(cfg, train_frame, val_frame, class_names, seed, device):
    seed_everything(seed)
    preprocessor = MetadataPreprocessor(cfg["data"]["metadata_features"]).fit(train_frame)
    require_images = cfg["model"]["type"] not in ("metadata_linear", "metadata_mlp")
    train_loader = make_loader(cfg, train_frame, preprocessor, True, require_images, seed)
    val_loader = make_loader(cfg, val_frame, preprocessor, False, require_images, seed)
    model, criterion, optimizer = build_training_objects(
        cfg, train_frame, preprocessor, class_names, device
    )

    best_score = -np.inf
    best_epoch = 1
    stale = 0
    for epoch in range(cfg["training"]["epochs"]):
        set_backbone_state(model, epoch < cfg["training"]["freeze_backbone_epochs"])
        common = {
            "alignment_weight": cfg["training"]["alignment_weight"],
            "alignment_temperature": cfg["training"]["alignment_temperature"],
            "amp": cfg["training"]["amp"],
        }
        run_epoch(model, train_loader, criterion, device, optimizer=optimizer, **common)
        val_metrics = run_epoch(model, val_loader, criterion, device, **common)
        score = val_metrics["balanced_accuracy"]
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= cfg["training"]["early_stopping_patience"]:
                break

    del model, optimizer, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return float(best_score), int(best_epoch)


def fit_outer_model(cfg, train_frame, test_frame, class_names, epochs, seed, device):
    seed_everything(seed)
    preprocessor = MetadataPreprocessor(cfg["data"]["metadata_features"]).fit(train_frame)
    require_images = cfg["model"]["type"] not in ("metadata_linear", "metadata_mlp")
    train_loader = make_loader(cfg, train_frame, preprocessor, True, require_images, seed)
    test_loader = make_loader(cfg, test_frame, preprocessor, False, require_images, seed)
    model, criterion, optimizer = build_training_objects(
        cfg, train_frame, preprocessor, class_names, device
    )

    for epoch in range(epochs):
        set_backbone_state(model, epoch < cfg["training"]["freeze_backbone_epochs"])
        run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            alignment_weight=cfg["training"]["alignment_weight"],
            alignment_temperature=cfg["training"]["alignment_temperature"],
            amp=cfg["training"]["amp"],
        )

    _, labels, probabilities = predict(model, test_loader, device)
    metrics = calculate_metrics(labels, probabilities, class_names)
    del model, optimizer, train_loader, test_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def summarize_importance(per_fold):
    full = per_fold.loc[per_fold["experiment"].eq("full")].set_index("outer_fold")
    importance_rows = []
    for ablation_name, input_name in INPUT_FROM_ABLATION.items():
        ablated = per_fold.loc[per_fold["experiment"].eq(ablation_name)].set_index("outer_fold")
        common_folds = full.index.intersection(ablated.index)
        for outer_fold in common_folds:
            row = {"input": input_name, "outer_fold": int(outer_fold)}
            for metric in IMPORTANCE_METRICS:
                row[f"{metric}_importance"] = float(
                    full.loc[outer_fold, metric] - ablated.loc[outer_fold, metric]
                )
            importance_rows.append(row)

    per_fold_importance = pd.DataFrame(importance_rows)
    if per_fold_importance.empty:
        return per_fold_importance, pd.DataFrame()
    summary_rows = []
    for input_name, group in per_fold_importance.groupby("input", sort=False):
        row = {"input": input_name, "outer_folds": len(group)}
        for metric in IMPORTANCE_METRICS:
            values = group[f"{metric}_importance"]
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_importance_mean"] = mean
            row[f"{metric}_importance_std"] = std
            row[f"{metric}_importance_mean_std"] = f"{mean:.4f} +/- {std:.4f}"
        summary_rows.append(row)
    return per_fold_importance, pd.DataFrame(summary_rows)


def performance_summary(per_fold):
    rows = []
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "auroc",
               "concerning_sensitivity", "concerning_specificity"]
    for experiment, group in per_fold.groupby("experiment", sort=False):
        row = {"experiment": experiment, "outer_folds": len(group)}
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_mean_std"] = f"{mean:.4f} +/- {std:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="5x5 nested CV input ablation study.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs/nested_cv_importance")
    parser.add_argument("--outer-folds", type=int)
    parser.add_argument("--inner-folds", type=int)
    parser.add_argument("--experiments", nargs="+", choices=list(ABLATIONS), default=list(ABLATIONS))
    parser.add_argument("--epochs", type=int, help="Override maximum epochs for development runs.")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg["data"].get("task", "binary") != "binary":
        raise ValueError("Input-importance nested CV currently requires data.task: binary.")
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
    class_names = task_classes("binary")
    frame = load_isic_tables(
        cfg["data"]["metadata_csv"],
        cfg["data"]["ground_truth_csv"],
        drop_unmatched=True,
        task="binary",
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
    assignment_frame = frame[["image", "diagnosis", "label", "lesion_id"]].copy()
    assignment_frame["outer_fold"] = outer_assignment
    assignment_frame.to_csv(output_dir / "outer_fold_assignments.csv", index=False)
    inner_rows = []
    selected_rows = []
    outer_rows = []

    for experiment_name in args.experiments:
        ablation = ABLATIONS[experiment_name]
        print(f"\n===== Experiment: {experiment_name} =====")
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
                candidate_cfg = experiment_config(cfg, ablation, candidate)
                for inner_index, (inner_train_idx, inner_val_idx) in enumerate(inner_splits, start=1):
                    inner_train = outer_train.iloc[inner_train_idx].reset_index(drop=True)
                    inner_val = outer_train.iloc[inner_val_idx].reset_index(drop=True)
                    # Use the same seed for paired ablations and candidates in a fold.
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
            selected_cfg = experiment_config(cfg, ablation, selected["config"])
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

    per_fold = pd.DataFrame(outer_rows)
    performance = performance_summary(per_fold)
    performance.to_csv(output_dir / "performance_summary.csv", index=False)
    importance_per_fold, importance = summarize_importance(per_fold)
    importance_per_fold.to_csv(output_dir / "input_importance_per_fold.csv", index=False)
    importance.to_csv(output_dir / "input_importance_summary.csv", index=False)
    (output_dir / "run_config.json").write_text(json.dumps({
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "experiments": args.experiments,
        "candidates": candidates,
        "fixed_cohort_size": len(frame),
        "importance_definition": "full outer-fold metric minus remove-input outer-fold metric",
    }, indent=2), encoding="utf-8")

    print("\nPerformance mean +/- std across outer folds:")
    print(performance.to_string(index=False))
    print("\nInput importance (positive means the input helped):")
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
