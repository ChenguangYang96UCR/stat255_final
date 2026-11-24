#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.isic_multimodal import task_classes
from src.isic_multimodal.data import (ISICDataset, MetadataPreprocessor, image_transforms,
                                      load_isic_tables)
from src.isic_multimodal.engine import predict, run_epoch
from src.isic_multimodal.losses import build_loss, class_weights
from src.isic_multimodal.metrics import calculate_metrics
from src.isic_multimodal.models import build_model
from src.isic_multimodal.utils import atomic_torch_save, device_from_environment, load_config, save_json, seed_everything


def make_loaders(cfg, frame, preprocessor):
    d = cfg["data"]
    metadata = preprocessor.transform(frame)
    loaders = {}
    require_images = cfg["model"]["type"] not in ("metadata_linear", "metadata_mlp")
    for split in ("train", "val", "test"):
        mask = frame["split"].eq(split).to_numpy()
        ds = ISICDataset(frame.loc[mask], d["image_dir"], metadata[mask],
                         image_transforms(d["image_size"], train=split == "train"), require_images)
        loaders[split] = DataLoader(ds, batch_size=d["batch_size"], shuffle=split == "train",
                                    num_workers=d["num_workers"], pin_memory=torch.cuda.is_available(),
                                    persistent_workers=d["num_workers"] > 0)
    return loaders


def main():
    parser = argparse.ArgumentParser(description="Train an ISIC 2019 unimodal or multimodal classifier.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--task", choices=["binary", "multiclass"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-type", choices=["metadata_linear", "metadata_mlp", "image", "multimodal"])
    parser.add_argument("--fusion", choices=["concat", "gated", "decision", "attention"])
    parser.add_argument("--loss", choices=["ce", "weighted_ce", "focal"])
    parser.add_argument("--alignment-weight", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--metadata-features", nargs="+",
                        choices=["age_approx", "sex", "anatom_site_general"])
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.task: cfg["data"]["task"] = args.task
    if args.seed is not None: cfg["seed"] = args.seed
    if args.model_type: cfg["model"]["type"] = args.model_type
    if args.fusion: cfg["model"]["fusion"] = args.fusion
    if args.loss: cfg["training"]["loss"] = args.loss
    if args.alignment_weight is not None: cfg["training"]["alignment_weight"] = args.alignment_weight
    if args.output_dir: cfg["output_dir"] = args.output_dir
    if args.metadata_features: cfg["data"]["metadata_features"] = args.metadata_features
    if args.no_pretrained: cfg["model"]["pretrained"] = False

    seed_everything(cfg["seed"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    task = cfg["data"].get("task", "multiclass")
    class_names = task_classes(task)
    frame = load_isic_tables(
        cfg["data"]["metadata_csv"],
        cfg["data"]["ground_truth_csv"],
        drop_unmatched=True,
        task=task,
    )
    splits = pd.read_csv(cfg["data"]["split_csv"])
    frame = frame.merge(splits[["image", "split"]], on="image", how="inner", validate="one_to_one")
    if len(frame) != len(splits):
        raise ValueError("Split CSV does not match the current dataset.")
    train_frame = frame[frame["split"] == "train"]
    preprocessor = MetadataPreprocessor(cfg["data"].get("metadata_features")).fit(train_frame)
    preprocessor.save(output_dir / "metadata_preprocessor.json")
    loaders = make_loaders(cfg, frame, preprocessor)

    device = device_from_environment()
    print(f"Using device: {device}")
    model = build_model(cfg, preprocessor.output_dim, len(class_names)).to(device)
    weights = class_weights(train_frame["label"].to_numpy(), len(class_names)).to(device)
    criterion_weights = weights if cfg["training"]["loss"] in ("weighted_ce", "focal") else None
    criterion = build_loss(cfg["training"]["loss"], criterion_weights, cfg["training"]["focal_gamma"])

    backbone_ids = set()
    if hasattr(model, "image_encoder"):
        backbone_ids = {id(p) for p in model.image_encoder.features.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    backbone_params = [p for p in model.parameters() if id(p) in backbone_ids]
    groups = [{"params": head_params, "lr": cfg["training"]["learning_rate"]}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": cfg["training"]["backbone_learning_rate"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["training"]["weight_decay"])

    history, best_score, best_epoch, stale = [], -1.0, -1, 0
    for epoch in range(cfg["training"]["epochs"]):
        if hasattr(model, "image_encoder"):
            model.image_encoder.freeze_backbone(epoch < cfg["training"]["freeze_backbone_epochs"])
        common = dict(alignment_weight=cfg["training"]["alignment_weight"],
                      alignment_temperature=cfg["training"]["alignment_temperature"], amp=cfg["training"]["amp"])
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer=optimizer, **common)
        val_metrics = run_epoch(model, loaders["val"], criterion, device, **common)
        row = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        history.append(row)
        score = val_metrics["balanced_accuracy"]
        print(f"Epoch {epoch + 1:02d}: train loss={train_metrics['loss']:.4f}, "
              f"val balanced_acc={score:.4f}, val macro_f1={val_metrics['macro_f1']:.4f}")
        if score > best_score:
            best_score, best_epoch, stale = score, epoch + 1, 0
            atomic_torch_save({"model": model.state_dict(), "config": cfg, "epoch": best_epoch,
                               "metadata_dim": preprocessor.output_dim, "task": task,
                               "class_names": class_names}, output_dir / "best.pt")
        else:
            stale += 1
            if stale >= cfg["training"]["early_stopping_patience"]:
                print("Early stopping.")
                break
    save_json(history, output_dir / "history.json")

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    ids, labels, probabilities = predict(model, loaders["test"], device)
    test_metrics = calculate_metrics(labels, probabilities, class_names)
    test_metrics["selected_epoch"] = best_epoch
    test_metrics["seed"] = cfg["seed"]
    test_metrics["task"] = task
    test_metrics["model_type"] = cfg["model"]["type"]
    test_metrics["fusion"] = cfg["model"].get("fusion")
    save_json(test_metrics, output_dir / "test_metrics.json")
    pred = pd.DataFrame({"image": ids, "label": labels, "prediction": probabilities.argmax(1)})
    for i, class_name in enumerate(class_names):
        pred[f"prob_{class_name}"] = probabilities[:, i]
    pred.to_csv(output_dir / "test_predictions.csv", index=False)
    plt.figure(figsize=(9, 7))
    sns.heatmap(test_metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close()
    print(f"Best epoch: {best_epoch}; test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
