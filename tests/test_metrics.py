import numpy as np

from src.isic_multimodal.metrics import calculate_metrics


def test_binary_metrics_use_positive_class_probability():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.1, 0.9],
    ])
    result = calculate_metrics(labels, probabilities, ["benign", "concerning"])
    assert result["accuracy"] == 1.0
    assert result["auroc"] == 1.0
    assert result["concerning_sensitivity"] == 1.0
    assert result["concerning_specificity"] == 1.0
