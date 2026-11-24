#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.isic_multimodal.data import load_isic_tables, make_splits


DEFAULT_REQUIRED_METADATA = ["age_approx", "anatom_site_general", "lesion_id", "sex"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def filter_complete_samples(frame, required_metadata, image_dir=None):
    """Keep rows with complete metadata and, when requested, an existing image."""
    missing_columns = [column for column in required_metadata if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Metadata CSV is missing required columns: {missing_columns}")

    valid_metadata = frame[required_metadata].notna().all(axis=1)
    for column in required_metadata:
        valid_metadata &= frame[column].astype("string").str.strip().ne("").fillna(False)

    filtered = frame.loc[valid_metadata].copy()
    removed_metadata = len(frame) - len(filtered)

    removed_images = 0
    if image_dir is not None:
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        image_ids = {
            path.stem
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        has_image = filtered["image"].astype(str).isin(image_ids)
        removed_images = int((~has_image).sum())
        filtered = filtered.loc[has_image].copy()

    return filtered.reset_index(drop=True), removed_metadata, removed_images


def main():
    parser = argparse.ArgumentParser(description="Filter ISIC 2019 data and create grouped splits.")
    parser.add_argument("--metadata", default="data/ISIC_2019_Training_Metadata.csv")
    parser.add_argument("--ground-truth", default="data/ISIC_2019_Training_GroundTruth.csv")
    parser.add_argument("--image-dir", default="data/ISIC_2019_Training_Input")
    parser.add_argument("--output", default="data/splits_binary.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument(
        "--required-metadata",
        nargs="+",
        default=DEFAULT_REQUIRED_METADATA,
        help="Metadata columns that must be non-missing for a sample to be included.",
    )
    parser.add_argument("--skip-image-check", action="store_true")
    args = parser.parse_args()

    frame = load_isic_tables(
        args.metadata,
        args.ground_truth,
        drop_unmatched=True,
        task=args.task,
    )
    dropped_unmatched = frame.attrs.get("dropped_unmatched_metadata", 0)
    if dropped_unmatched:
        examples = frame.attrs.get("unmatched_metadata_examples", [])
        print(
            f"Removed {dropped_unmatched} ground-truth samples without a metadata row "
            f"(examples: {examples})."
        )
    frame, removed_metadata, removed_images = filter_complete_samples(
        frame,
        args.required_metadata,
        None if args.skip_image_check else args.image_dir,
    )
    if frame.empty:
        raise ValueError("No samples remain after filtering metadata and images.")

    print(f"Removed {removed_metadata} samples with missing required metadata.")
    if args.skip_image_check:
        print("Skipped image availability filtering.")
    else:
        print(f"Removed {removed_images} samples without a matching image file.")
    print(f"Keeping {len(frame)} samples for splitting.\n")
    print(f"Task: {args.task}")

    splits = make_splits(frame, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.output, index=False)
    print(splits["split"].value_counts().sort_index().to_string())
    print("\nLabel counts by split:")
    print(splits.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()

