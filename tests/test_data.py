import pandas as pd
import pytest

from src.isic_multimodal import CLASSES
from src.isic_multimodal.data import MetadataPreprocessor, load_isic_tables, make_splits
from prepare_data import filter_complete_samples


def test_metadata_preprocessor_handles_missing_and_unknown():
    train = pd.DataFrame({"age_approx": [20, None, 60], "sex": ["male", "female", None],
                          "anatom_site_general": ["head/neck", "torso", "torso"]})
    processor = MetadataPreprocessor().fit(train)
    test = pd.DataFrame({"age_approx": [None], "sex": ["other"], "anatom_site_general": [None]})
    transformed = processor.transform(test)
    assert transformed.shape == (1, processor.output_dim)
    assert transformed[0, 1] == 1.0


def test_group_split_has_no_leakage():
    rows = []
    for label in range(8):
        for group in range(8):
            for image in range(2):
                rows.append({"image": f"{label}_{group}_{image}", "label": label,
                             "lesion_id": f"lesion_{label}_{group}"})
    split = make_splits(pd.DataFrame(rows), seed=1)
    assert set(split["split"]) == {"train", "val", "test"}
    assert split.groupby("group")["split"].nunique().max() == 1


def test_prepare_data_filters_missing_metadata_and_unmatched_images(tmp_path):
    frame = pd.DataFrame({
        "image": ["keep", "missing_age", "blank_sex", "missing_file"],
        "label": [0, 1, 2, 3],
        "age_approx": [50, None, 40, 60],
        "anatom_site_general": ["torso", "head/neck", "torso", "leg"],
        "lesion_id": ["a", "b", "c", "d"],
        "sex": ["female", "male", "  ", "male"],
    })
    (tmp_path / "keep.jpg").touch()

    filtered, removed_metadata, removed_images = filter_complete_samples(
        frame,
        ["age_approx", "anatom_site_general", "lesion_id", "sex"],
        tmp_path,
    )

    assert filtered["image"].tolist() == ["keep"]
    assert removed_metadata == 2
    assert removed_images == 1


def test_load_tables_can_drop_ground_truth_rows_without_metadata(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    truth_path = tmp_path / "truth.csv"
    pd.DataFrame({
        "image": ["keep"],
        "age_approx": [50],
        "anatom_site_general": ["torso"],
        "lesion_id": ["lesion_1"],
        "sex": ["female"],
    }).to_csv(metadata_path, index=False)

    truth = pd.DataFrame(0, index=range(2), columns=CLASSES)
    truth.insert(0, "image", ["keep", "no_metadata"])
    truth.loc[0, CLASSES[0]] = 1
    truth.loc[1, CLASSES[1]] = 1
    truth.to_csv(truth_path, index=False)

    with pytest.raises(ValueError, match="Images without metadata"):
        load_isic_tables(metadata_path, truth_path)

    filtered = load_isic_tables(metadata_path, truth_path, drop_unmatched=True)
    assert filtered["image"].tolist() == ["keep"]
    assert filtered.attrs["dropped_unmatched_metadata"] == 1


def test_binary_labels_map_benign_and_concerning_diagnoses(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    truth_path = tmp_path / "truth.csv"
    image_ids = [f"sample_{name}" for name in CLASSES]
    pd.DataFrame({
        "image": image_ids,
        "age_approx": [50] * len(CLASSES),
        "anatom_site_general": ["torso"] * len(CLASSES),
        "lesion_id": [f"lesion_{i}" for i in range(len(CLASSES))],
        "sex": ["female"] * len(CLASSES),
    }).to_csv(metadata_path, index=False)

    truth = pd.DataFrame(0, index=range(len(CLASSES)), columns=CLASSES)
    truth.insert(0, "image", image_ids)
    for index, name in enumerate(CLASSES):
        truth.loc[index, name] = 1
    truth.to_csv(truth_path, index=False)

    frame = load_isic_tables(metadata_path, truth_path, task="binary")
    labels = dict(zip(frame["diagnosis"], frame["label"]))
    assert {name for name, label in labels.items() if label == 0} == {"NV", "BKL", "DF", "VASC"}
    assert {name for name, label in labels.items() if label == 1} == {"MEL", "BCC", "SCC", "AK"}

