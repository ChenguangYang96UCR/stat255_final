# ISIC 2019 multimodal skin-lesion classification

This repository implements metadata-only, image-only, and multimodal EfficientNet-B0 classifiers. The default task is binary classification:

- `0 = benign`: `NV`, `BKL`, `DF`, `VASC`
- `1 = concerning`: `MEL`, `BCC`, `SCC`, `AK`

The name `concerning` is used instead of `unhealthy` because every ISIC image contains a skin lesion and `AK` is a precancerous lesion. The original eight-class task is still available with `--task multiclass`.

## 1. Put the official data in place

Download the labeled ISIC 2019 training input, ground truth, and metadata archives. After extraction, the default layout is:

```text
data/
├── ISIC_2019_Training_Input/
│   ├── ISIC_0000000.jpg
│   └── ...
├── ISIC_2019_Training_GroundTruth.csv
└── ISIC_2019_Training_Metadata.csv
```

The code expects the eight training classes `MEL`, `NV`, `BCC`, `AK`, `BKL`, `DF`, `VASC`, and `SCC`. It intentionally does not train on the challenge test set's outlier class.

## 2. Create the environment

Python 3.10–3.12 is recommended because PyTorch support for the newest Python release can lag.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On a CUDA machine, install the matching PyTorch build from the official PyTorch selector before installing the remaining requirements.

## 3. Validate data and create the fixed split

```bash
python prepare_data.py
```

This joins the tables by `image` and filters samples while loading. Ground-truth rows with no matching metadata row are skipped. By default, a remaining sample is kept only when `age_approx`, `anatom_site_general`, `lesion_id`, and `sex` are all present and its image file exists. The original CSV and image directory are not changed; only valid image IDs are written to `data/splits.csv`. The split groups samples by `lesion_id`, then `patient_id`, when available, and otherwise falls back to the image identifier. The same split file is reused by every experiment.

To require a different set of metadata fields, pass them explicitly:

```bash
python prepare_data.py --required-metadata age_approx sex anatom_site_general
```

For the original eight-class task, create a separate split file:

```bash
python prepare_data.py --task multiclass --output data/splits_multiclass.csv
```

## 4. Train models

Edit `config.yaml` if needed, then run one model:

```bash
# Metadata logistic-regression-style linear baseline
python train.py --model-type metadata_linear --output-dir outputs/metadata_linear

# Metadata MLP
python train.py --model-type metadata_mlp --output-dir outputs/metadata_mlp

# Image-only EfficientNet-B0
python train.py --model-type image --output-dir outputs/image

# Multimodal concatenation
python train.py --model-type multimodal --fusion concat --output-dir outputs/concat

# Gated fusion with contrastive alignment
python train.py --model-type multimodal --fusion gated --alignment-weight 0.1 --output-dir outputs/gated_align

# Metadata ablation: image plus age only
python train.py --model-type multimodal --fusion concat \
  --metadata-features age_approx --output-dir outputs/concat_age
```

If the machine cannot download ImageNet weights, first obtain them in a networked environment or add `--no-pretrained` for a pipeline test. Training from scratch is not the proposal's intended final experiment.

To run the full controlled comparison sequentially:

```bash
python run_experiments.py
```

`config.yaml` is set to `data.task: binary` and `data/splits_binary.csv`. To train the eight-class version, change both settings to `multiclass` and `data/splits_multiclass.csv`, or pass `--task multiclass` while using the matching split file.

For a shorter initial check:

```bash
python run_experiments.py --quick
```

## 5. Outputs and comparison

Each run writes the best validation-selected checkpoint, preprocessing state, training history, test metrics, test predictions, and confusion matrix under its output directory. The test set is evaluated only after early stopping has selected the best epoch by validation balanced accuracy.

Compare two models with a paired bootstrap confidence interval:

```bash
python compare_models.py \
  outputs/image/test_predictions.csv \
  outputs/concat/test_predictions.csv \
  --metric balanced_accuracy --samples 2000
```

## Useful configuration changes

- Set `training.loss` to `ce`, `weighted_ce`, or `focal`.
- Set `training.alignment_weight` to `0.0` for the no-alignment baseline.
- Reduce `data.batch_size` if GPU memory is limited.
- Set `data.num_workers: 0` on platforms where multiprocessing data loading is problematic.
- Set `model.decision_alpha` using validation experiments only.

## Tests

```bash
python -m pytest -q
```

The tests cover load-time metadata/image filtering, metadata preprocessing, and group leakage in the split.
