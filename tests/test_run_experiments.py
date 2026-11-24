import json

import pytest

from run_experiments import aggregate_results


def test_aggregate_results_reports_sample_mean_and_std(tmp_path):
    values = [0.6, 0.7, 0.8]
    for seed, value in zip((42, 43, 44), values):
        result_dir = tmp_path / "concat" / f"seed_{seed}"
        result_dir.mkdir(parents=True)
        (result_dir / "test_metrics.json").write_text(
            json.dumps({"balanced_accuracy": value, "macro_f1": value - 0.1}),
            encoding="utf-8",
        )

    per_seed, summary = aggregate_results(tmp_path, ["concat"], [42, 43, 44])
    assert len(per_seed) == 3
    assert summary.loc[0, "balanced_accuracy_mean"] == pytest.approx(0.7)
    assert summary.loc[0, "balanced_accuracy_std"] == pytest.approx(0.1)
    assert summary.loc[0, "balanced_accuracy_mean_std"] == "0.7000 +/- 0.1000"
    assert (tmp_path / "per_seed_results.csv").exists()
    assert (tmp_path / "summary_mean_std.csv").exists()
    assert (tmp_path / "summary_mean_std.json").exists()
