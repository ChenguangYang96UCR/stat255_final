#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.isic_multimodal.data import load_isic_tables, make_splits


def main():
    parser = argparse.ArgumentParser(description="Validate ISIC 2019 tables and create grouped splits.")
    parser.add_argument("--metadata", default="data/ISIC_2019_Training_Metadata.csv")
    parser.add_argument("--ground-truth", default="data/ISIC_2019_Training_GroundTruth.csv")
    parser.add_argument("--image-dir", default="data/ISIC_2019_Training_Input")
    parser.add_argument("--output", default="data/splits.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-image-check", action="store_true")
    args = parser.parse_args()

    frame = load_isic_tables(args.metadata, args.ground_truth)
    if not args.skip_image_check:
        image_dir = Path(args.image_dir)
        stems = {p.stem for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG") for p in image_dir.glob(pattern)}
        missing = sorted(set(frame["image"]) - stems)
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} images; first examples: {missing[:10]}")
    splits = make_splits(frame, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.output, index=False)
    print(splits["split"].value_counts().sort_index().to_string())
    print("\nClass counts by split:")
    print(splits.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()

