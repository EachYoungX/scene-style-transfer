import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_compatibility_labels_cover_all_diagnostic_cases():
    with (ROOT / "configs/experiment/compatibility_diagnostic_pairs.csv").open(newline="", encoding="utf-8") as f:
        manifest_cases = {row["case_id"] for row in csv.DictReader(f)}
    with (ROOT / "configs/experiment/compatibility_sweep_labels.csv").open(newline="", encoding="utf-8") as f:
        label_cases = {row["case_id"] for row in csv.DictReader(f)}

    assert manifest_cases <= label_cases


def test_compatibility_labels_include_failure_modes():
    with (ROOT / "configs/experiment/compatibility_sweep_labels.csv").open(newline="", encoding="utf-8") as f:
        modes = {row["observed_mode"] for row in csv.DictReader(f)}

    assert "reference_takeover" in modes
    assert "structural_coercion" in modes
    assert "stable_stylization_then_regeneration" in modes
