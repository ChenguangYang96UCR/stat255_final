import pandas as pd

from src.isic_multimodal.data import MetadataPreprocessor, make_splits


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

