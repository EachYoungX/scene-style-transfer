"""Normalize and audit V2.1 Subject/Background annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.append(str(ROOT / "src"))

from regions.v2_1_masks import load_region_mask_set  # noqa: E402


CONTENT_NAMES = {
    "v1_5_demuth_church": "photo_church.png",
    "v1_5_kulhanek_snow_winter": "photo_snow_winter.png",
    "v1_5_demuth_wave": "photo_wave.png",
}


def save_mask(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array.astype(np.uint8) * 255, mode="L").save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--case-id", action="append")
    args = parser.parse_args()
    source_root = ROOT / "data/derived/v2_0_geometry_risk/annotations/soft_stylization"
    rigid_root = ROOT / "data/derived/v2_0_geometry_risk/annotations/rigid_structure"
    valid_content_root = ROOT / "data/derived/v2_0_geometry_risk/valid_masks/valid_content"
    valid_eval_root = ROOT / "data/derived/v2_0_geometry_risk/valid_masks/valid_eval"
    output_root = ROOT / "data/derived/v2_1_geometry_control/region_masks"
    case_ids = args.case_id or list(CONTENT_NAMES)
    reports = {}
    for case_id in case_ids:
        content_name = CONTENT_NAMES[case_id]
        stem = Path(content_name).stem
        masks = load_region_mask_set(
            source_root / f"{stem}_S.png",
            source_root / f"{stem}_B.png",
            rigid_root / content_name,
            valid_content_root / content_name,
            valid_eval_root / content_name,
            threshold=args.threshold,
        )
        case_root = output_root / stem
        case_root.mkdir(parents=True, exist_ok=True)
        for name, value in {
            "subject_raw": masks.subject_raw,
            "background_raw": masks.background_raw,
            "subject": masks.subject,
            "background": masks.background,
            "neutral": masks.neutral,
            "rigid_excluded": masks.rigid & masks.valid_eval,
            "valid_eval": masks.valid_eval,
        }.items():
            save_mask(value, case_root / f"{name}.png")
        report = masks.report()
        report.update(
            {
                "case_id": case_id,
                "source_subject": str((source_root / f"{stem}_S.png").relative_to(ROOT)),
                "source_background": str((source_root / f"{stem}_B.png").relative_to(ROOT)),
                "threshold": args.threshold,
                "overlap_policy": "subject_priority_then_rigid_exclusion",
                "has_manual_neutral": False,
                "effective_partition_scope": "valid_eval_without_rigid",
            }
        )
        (case_root / "mask_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports[case_id] = report
        print(json.dumps(report, indent=2))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps({"threshold": args.threshold, "cases": reports}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
