import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rigidity_diagnostic_manifest_has_two_cases_per_group():
    with (ROOT / "configs/experiment/rigidity_diagnostic_pairs.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    counts = {}
    for row in rows:
        counts[row["scene_group"]] = counts.get(row["scene_group"], 0) + 1

    assert counts == {"A": 2, "B": 2, "C": 2, "D": 2}


def test_rigidity_diagnostic_files_exist():
    with (ROOT / "configs/experiment/rigidity_diagnostic_pairs.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        assert (ROOT / row["content"]).exists()
        assert (ROOT / row["style"]).exists()
