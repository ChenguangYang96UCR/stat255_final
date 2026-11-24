import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import Dataset

from . import CLASSES


def load_isic_tables(metadata_csv, ground_truth_csv):
    meta = pd.read_csv(metadata_csv)
    truth = pd.read_csv(ground_truth_csv)
    if "image" not in meta or "image" not in truth:
        raise ValueError("Both CSV files must contain an 'image' column.")
    missing = [c for c in CLASSES if c not in truth.columns]
    if missing:
        raise ValueError(f"Ground-truth CSV is missing class columns: {missing}")
    if truth[CLASSES].isna().any().any():
        raise ValueError("Ground-truth class columns contain missing values.")
    row_sums = truth[CLASSES].sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError("Each ground-truth row must contain exactly one positive class.")
    truth = truth.copy()
    truth["label"] = truth[CLASSES].to_numpy().argmax(axis=1)
    if meta["image"].duplicated().any() or truth["image"].duplicated().any():
        raise ValueError("Duplicate image identifiers found in an input CSV.")
    df = truth[["image", "label"]].merge(meta, on="image", how="left", validate="one_to_one")
    if df[[c for c in meta.columns if c != "image"]].isna().all(axis=1).any():
        bad = df.loc[df[[c for c in meta.columns if c != "image"]].isna().all(axis=1), "image"]
        raise ValueError(f"Images without metadata: {bad.head().tolist()}")
    return df


def _group_column(df):
    for col in ("lesion_id", "patient_id"):
        if col in df and df[col].notna().any():
            # Missing group ids must not collapse unrelated rows into one group.
            return df[col].fillna("missing_" + df["image"].astype(str))
    return df["image"]


def make_splits(df, seed=42):
    """Approximate 70/15/15 split while keeping each group in one partition."""
    y = df["label"].to_numpy()
    groups = _group_column(df).astype(str).to_numpy()
    try:
        outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
        train_val_idx, test_idx = next(outer.split(df, y, groups))
        inner_df = df.iloc[train_val_idx]
        inner_y = y[train_val_idx]
        inner_groups = groups[train_val_idx]
        inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed + 1)
        train_rel, val_rel = next(inner.split(inner_df, inner_y, inner_groups))
        train_idx = train_val_idx[train_rel]
        val_idx = train_val_idx[val_rel]
    except ValueError:
        # Small development datasets may not have enough groups per class.
        first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        train_val_idx, test_idx = next(first.split(df, y, groups))
        second = GroupShuffleSplit(n_splits=1, test_size=0.17647, random_state=seed + 1)
        tr_rel, val_rel = next(second.split(df.iloc[train_val_idx], y[train_val_idx], groups[train_val_idx]))
        train_idx, val_idx = train_val_idx[tr_rel], train_val_idx[val_rel]
    split = np.full(len(df), "", dtype=object)
    split[train_idx], split[val_idx], split[test_idx] = "train", "val", "test"
    result = df[["image", "label"]].copy()
    result["split"] = split
    result["group"] = groups
    assert not result.groupby("group")["split"].nunique().gt(1).any()
    return result


class MetadataPreprocessor:
    def __init__(self, features=None):
        self.features = features or ["age_approx", "sex", "anatom_site_general"]
        self.age_median = 0.0
        self.age_mean = 0.0
        self.age_std = 1.0
        self.categories = {}
        self.categorical_columns = []

    def fit(self, df):
        if "age_approx" in self.features:
            age = pd.to_numeric(df.get("age_approx", pd.Series(np.nan, index=df.index)), errors="coerce")
            self.age_median = float(age.median()) if age.notna().any() else 0.0
            filled = age.fillna(self.age_median)
            self.age_mean = float(filled.mean())
            self.age_std = float(filled.std(ddof=0)) or 1.0
        self.categorical_columns = [c for c in ("sex", "anatom_site_general")
                                    if c in self.features and c in df.columns]
        for col in self.categorical_columns:
            vals = df[col].fillna("Unknown").astype(str)
            cats = sorted(set(vals.tolist()) | {"Unknown"})
            self.categories[col] = cats
        return self

    @property
    def output_dim(self):
        return (2 if "age_approx" in self.features else 0) + sum(
            len(self.categories[c]) for c in self.categorical_columns)

    def transform(self, df):
        arrays = []
        if "age_approx" in self.features:
            age = pd.to_numeric(df.get("age_approx", pd.Series(np.nan, index=df.index)), errors="coerce")
            age_missing = age.isna().astype(np.float32).to_numpy()[:, None]
            age_z = ((age.fillna(self.age_median) - self.age_mean) / self.age_std).astype(np.float32).to_numpy()[:, None]
            arrays.extend([age_z, age_missing])
        for col in self.categorical_columns:
            values = df[col].fillna("Unknown").astype(str)
            known = set(self.categories[col])
            values = values.where(values.isin(known), "Unknown")
            arr = np.zeros((len(df), len(self.categories[col])), dtype=np.float32)
            lookup = {v: i for i, v in enumerate(self.categories[col])}
            for row, value in enumerate(values):
                arr[row, lookup[value]] = 1.0
            arrays.append(arr)
        if not arrays:
            raise ValueError("At least one available metadata feature must be selected.")
        return np.concatenate(arrays, axis=1).astype(np.float32)

    def save(self, path):
        payload = {"features": self.features, "age_median": self.age_median, "age_mean": self.age_mean,
                   "age_std": self.age_std, "categories": self.categories,
                   "categorical_columns": self.categorical_columns}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        obj = cls()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        obj.__dict__.update(payload)
        return obj


def image_transforms(image_size=224, train=False):
    from torchvision import transforms

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(), normalize,
        ])
    return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize])


class ISICDataset(Dataset):
    def __init__(self, frame, image_dir, metadata, transform, require_images=True):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.metadata = metadata
        self.transform = transform
        self.require_images = require_images

    def __len__(self):
        return len(self.frame)

    def _image_path(self, image_id):
        for suffix in (".jpg", ".jpeg", ".png", ".JPG"):
            path = self.image_dir / f"{image_id}{suffix}"
            if path.exists():
                return path
        raise FileNotFoundError(f"No image file found for {image_id} in {self.image_dir}")

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        if self.require_images:
            with Image.open(self._image_path(row["image"])) as im:
                image = self.transform(im.convert("RGB"))
        else:
            image = torch.empty(0)
        return {
            "image": image,
            "metadata": torch.from_numpy(self.metadata[idx]),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "image_id": str(row["image"]),
        }
