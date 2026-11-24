"""ISIC 2019 multimodal classification package."""

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
BINARY_CLASSES = ["benign", "concerning"]
BENIGN_DIAGNOSES = {"NV", "BKL", "DF", "VASC"}
CONCERNING_DIAGNOSES = {"MEL", "BCC", "SCC", "AK"}


def task_classes(task):
    if task == "multiclass":
        return CLASSES
    if task == "binary":
        return BINARY_CLASSES
    raise ValueError(f"Unknown classification task: {task}")

