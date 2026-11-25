import pandas as pd
import pytest

from nested_cv_importance import grouped_folds, summarize_importance


def test_grouped_outer_folds_keep_lesions_separate():
    rows = []
    for group in range(20):
        for image in range(2):
            rows.append({
                "image": f"image_{group}_{image}",
                "lesion_id": f"lesion_{group}",
                "label": group % 2,
            })
    frame = pd.DataFrame(rows)
    folds = grouped_folds(frame, n_splits=5, seed=42)
    assert len(folds) == 5
    for train_index, test_index in folds:
        train_groups = set(frame.iloc[train_index]["lesion_id"])
        test_groups = set(frame.iloc[test_index]["lesion_id"])
        assert train_groups.isdisjoint(test_groups)


def test_input_importance_is_paired_full_minus_ablation():
    rows = []
    for fold in range(1, 6):
        rows.append({
            "experiment": "full",
            "outer_fold": fold,
            "balanced_accuracy": 0.80,
            "macro_f1": 0.78,
            "auroc": 0.90,
        })
        rows.append({
            "experiment": "no_age",
            "outer_fold": fold,
            "balanced_accuracy": 0.75,
            "macro_f1": 0.74,
            "auroc": 0.88,
        })
    per_fold, summary = summarize_importance(pd.DataFrame(rows))
    assert len(per_fold) == 5
    age = summary.loc[summary["input"].eq("age_approx")].iloc[0]
    assert age["balanced_accuracy_importance_mean"] == pytest.approx(0.05)
    assert age["macro_f1_importance_mean"] == pytest.approx(0.04)
    assert age["auroc_importance_mean"] == pytest.approx(0.02)
