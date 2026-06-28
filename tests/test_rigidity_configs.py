import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rigidity_groups_include_low_and_high_rigidity():
    with (ROOT / "configs/experiment/rigidity_groups.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    levels = {row["rigidity_level"] for row in rows}
    groups = {row["group_id"] for row in rows}

    assert {"A", "B", "C", "D"} <= groups
    assert "low" in levels
    assert "high" in levels


def test_twopass_labels_cover_forest_and_architecture():
    with (ROOT / "configs/experiment/twopass_diagnostic_labels.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_case = {row["case_id"]: row for row in rows}

    assert by_case["debug_forest"]["scene_group"] == "B"
    assert by_case["debug_city_architecture"]["scene_group"] == "D"
