import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_compatibility_manifest_has_required_groups_and_files():
    with (ROOT / "configs/experiment/compatibility_diagnostic_pairs.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups = {row["compat_group"] for row in rows}
    assert {"G1", "G2", "G3", "G4"} <= groups

    for row in rows:
        assert (ROOT / row["content"]).exists()
        assert (ROOT / row["style"]).exists()


def test_compatibility_manifest_contains_new_samples():
    with (ROOT / "configs/experiment/compatibility_diagnostic_pairs.csv").open(newline="", encoding="utf-8") as f:
        text = f.read()

    assert "photo_sea_wave.jpg" in text
    assert "photo_flower_bed.jpg" in text
    assert "photo_lecreusois_church.jpg" in text
    assert "photo_seregei_city.jpg" in text
