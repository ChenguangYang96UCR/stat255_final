from nested_cv_classification import CLASSIFICATION_EXPERIMENTS, classification_config


def test_classification_config_applies_model_and_candidate_settings():
    base = {
        "data": {"metadata_features": []},
        "model": {"type": "image", "fusion": "concat", "dropout": 0.3},
        "training": {
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "alignment_weight": 0.0,
        },
    }
    candidate = {"learning_rate": 1e-4, "weight_decay": 2e-4, "dropout": 0.4}
    cfg = classification_config(base, CLASSIFICATION_EXPERIMENTS["gated_align"], candidate)
    assert cfg["model"]["type"] == "multimodal"
    assert cfg["model"]["fusion"] == "gated"
    assert cfg["model"]["dropout"] == 0.4
    assert cfg["training"]["learning_rate"] == 1e-4
    assert cfg["training"]["weight_decay"] == 2e-4
    assert cfg["training"]["alignment_weight"] == 0.1
    assert cfg["data"]["metadata_features"] == [
        "age_approx", "sex", "anatom_site_general"
    ]
