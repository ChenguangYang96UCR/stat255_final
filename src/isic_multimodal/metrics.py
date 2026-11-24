import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
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


def calculate_metrics(y_true, probabilities, class_names=None):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    class_names = list(class_names or CLASSES)
    num_classes = len(class_names)
    if probabilities.ndim != 2 or probabilities.shape[1] != num_classes:
        raise ValueError(
            f"Expected probabilities with {num_classes} columns, got {probabilities.shape}."
        )
    y_pred = probabilities.argmax(axis=1)
    labels = np.arange(num_classes)
    if num_classes == 2:
        try:
            auroc = float(roc_auc_score(y_true, probabilities[:, 1]))
        except ValueError:
            auroc = None
        brier = float(brier_score_loss(y_true, probabilities[:, 1]))
    else:
        one_hot = label_binarize(y_true, classes=labels)
        try:
            auroc = float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr"))
        except ValueError:
            auroc = None
        brier = float(np.mean(np.sum((one_hot - probabilities) ** 2, axis=1)))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auroc": auroc,
        "brier": brier,
        "ece": expected_calibration_error(y_true, probabilities),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if num_classes == 2:
        result["concerning_sensitivity"] = float(report[class_names[1]]["recall"])
        result["concerning_specificity"] = float(report[class_names[0]]["recall"])
    elif "MEL" in report:
        result["melanoma_sensitivity"] = float(report["MEL"]["recall"])
    return result
