"""Prepare a method-blinded absolute-scoring package from audited outputs."""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_RECORDS = ROOT / "external_benchmark/evaluation/automatic/output_audit_records.csv"
PAIR_MANIFEST = ROOT / "external_benchmark/configs/external_eval_pairs_v1.csv"
BLIND_ROOT = ROOT / "external_benchmark/evaluation/human_blind"
ASSET_ROOT = BLIND_ROOT / "assets"
SEED = 20260825

REVIEW_FIELDS = [
    "blind_id",
    "content_image",
    "reference_image",
    "output_image",
    "content_preservation",
    "style_fidelity",
    "geometry_takeover",
    "reference_semantic_leakage",
    "scene_regeneration",
    "new_reference_object",
    "false_hard_edge",
    "style_valid",
    "notes",
]
MAPPING_FIELDS = ["blind_id", "track_id", "method", "pair_id", "seed", "output_path"]


def hardlink_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        import shutil

        shutil.copy2(source, target)


def main() -> None:
    with PAIR_MANIFEST.open(newline="", encoding="utf-8") as handle:
        pairs = {row["pair_id"]: row for row in csv.DictReader(handle)}
    with AUDIT_RECORDS.open(newline="", encoding="utf-8") as handle:
        records = [row for row in csv.DictReader(handle) if row["audit_status"] == "pass"]
    random.Random(SEED).shuffle(records)

    BLIND_ROOT.mkdir(parents=True, exist_ok=True)
    mappings = []
    template_rows = []
    for index, record in enumerate(records, start=1):
        blind_id = f"blind_{index:04d}"
        pair = pairs[record["pair_id"]]
        asset_dir = ASSET_ROOT / blind_id
        output_source = ROOT / record["output_path"]
        content_source = ROOT / pair["content_path"]
        reference_source = ROOT / pair["reference_path"]
        hardlink_or_copy(content_source, asset_dir / "content.png")
        hardlink_or_copy(reference_source, asset_dir / "reference.png")
        hardlink_or_copy(output_source, asset_dir / "output.png")
        template_rows.append(
            {
                "blind_id": blind_id,
                "content_image": f"assets/{blind_id}/content.png",
                "reference_image": f"assets/{blind_id}/reference.png",
                "output_image": f"assets/{blind_id}/output.png",
                "content_preservation": "",
                "style_fidelity": "",
                "geometry_takeover": "",
                "reference_semantic_leakage": "",
                "scene_regeneration": "",
                "new_reference_object": "",
                "false_hard_edge": "",
                "style_valid": "",
                "notes": "",
            }
        )
        mappings.append(
            {
                "blind_id": blind_id,
                "track_id": record["track_id"],
                "method": record["method"],
                "pair_id": record["pair_id"],
                "seed": record["seed"],
                "output_path": record["output_path"],
            }
        )

    with (BLIND_ROOT / "review_template.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(template_rows)
    with (BLIND_ROOT / "analyst_mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAPPING_FIELDS)
        writer.writeheader()
        writer.writerows(mappings)
    (BLIND_ROOT / "README.md").write_text(
        """# External benchmark blind review

Fill `review_template.csv` only. Do not add method names or inspect `analyst_mapping.csv` during scoring.

Scores: `content_preservation` and `style_fidelity` use 0-4 (higher is better); `geometry_takeover`, `reference_semantic_leakage`, and `scene_regeneration` use 0-3 (lower is better). Binary fields use 0/1. Review each row independently with content, reference, and output visible.
""",
        encoding="utf-8",
    )
    print(f"blind_rows={len(template_rows)}")
    print(BLIND_ROOT / "review_template.csv")


if __name__ == "__main__":
    main()
