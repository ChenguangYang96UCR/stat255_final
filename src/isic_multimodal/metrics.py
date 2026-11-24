import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

from . import CLASSES


def expected_calibration_error(y_true, probabilities, bins=15):
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def calculate_metrics(y_true, probabilities):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    y_pred = probabilities.argmax(axis=1)
    one_hot = label_binarize(y_true, classes=np.arange(len(CLASSES)))
    try:
        auroc = float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr"))
    except ValueError:
        auroc = None
    brier = float(np.mean(np.sum((one_hot - probabilities) ** 2, axis=1)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_ovr_auroc": auroc,
        "multiclass_brier": brier,
        "ece": expected_calibration_error(y_true, probabilities),
        "melanoma_sensitivity": float(classification_report(
            y_true, y_pred, labels=np.arange(len(CLASSES)), target_names=CLASSES,
            output_dict=True, zero_division=0)["MEL"]["recall"]),
        "classification_report": classification_report(
            y_true, y_pred, labels=np.arange(len(CLASSES)), target_names=CLASSES,
            output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASSES))).tolist(),
    }
