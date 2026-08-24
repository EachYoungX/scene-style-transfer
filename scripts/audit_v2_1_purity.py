"""Pure-data V2.1 occupancy/purity audit; no model inference or result images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "src"), str(ROOT / "scripts")])

from regions.v2_1_purity import build_purity_routes, route_gate, save_purity_overlay
from run_baseline import fit_square
from run_v2_0_rigid_only import aligned_content_path, read_cases, select_case
from run_v2_1_regional_pilot import CONTENT_NAMES, region_paths
from regions.v2_1_masks import load_region_mask_set


CASE_IDS = ("v1_5_demuth_church", "v1_5_kulhanek_snow_winter", "v1_5_demuth_wave")
RESOLUTIONS = (64, 32, 16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/v2_1_regional_pilot.yaml")
    parser.add_argument("--output-dir", default="runs/ip_adapter_plus_injection/v2_1_purity_audit")
    parser.add_argument("--purity-threshold", type=float, default=0.8)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    route_rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    cases = read_cases(ROOT / config["experiment"]["manifest"])

    for case_id in CASE_IDS:
        case = select_case(cases, case_id, 42)
        name = CONTENT_NAMES[case_id]
        subject_path, background_path, rigid_path, valid_content_path, valid_eval_path = region_paths(
            ROOT, case_id, config
        )
        masks = load_region_mask_set(
            subject_path,
            background_path,
            rigid_path,
            valid_content_path,
            valid_eval_path,
            threshold=int(config["region_masks"]["threshold"]),
        )
        content = fit_square(aligned_content_path(ROOT, case), 512)
        case_output = output_dir / Path(name).stem
        case_output.mkdir(parents=True, exist_ok=True)
        routes = build_purity_routes(
            masks.subject,
            masks.background,
            masks.valid_eval,
            resolutions=RESOLUTIONS,
            purity_threshold=args.purity_threshold,
        )
        case_check = {
            "case": case_id,
            "pixel_overlap": masks.effective_overlap,
            "subject_pixels": int(masks.subject.sum()),
            "background_pixels": int(masks.background.sum()),
            "valid_eval_pixels": int(masks.valid_eval.sum()),
            "classes_partition_valid": True,
        }
        for resolution, purity in routes.items():
            save_purity_overlay(content, purity, case_output / f"purity_{resolution}x{resolution}.png")
            report = purity.report()
            rows.append({"case": case_id, **report})
            valid_count = report["valid_tokens"]
            class_count = report["pure_subject"] + report["mixed"] + report["pure_background"]
            if class_count != valid_count:
                case_check["classes_partition_valid"] = False
            for strategy in ("S_sep_neutral", "S_sep_conservative"):
                gate = route_gate(purity, strategy)
                route_rows.append(
                    {
                        "case": case_id,
                        "resolution": resolution,
                        "strategy": strategy,
                        "pure_subject_tokens": report["pure_subject"],
                        "mixed_tokens": report["mixed"],
                        "pure_background_tokens": report["pure_background"],
                        "mixed_gain": float(gate[purity.mixed][0]) if purity.mixed.any() else 0.0,
                        "max_gain": float(gate.max()),
                        "mixed_not_subject_amplified": bool(np.all(gate[purity.mixed] <= 1.0)),
                    }
                )
        checks.append(case_check)

    def write_csv(name: str, values: list[dict[str, object]]) -> None:
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

    write_csv("purity_counts.csv", rows)
    write_csv("route_checks.csv", route_rows)
    (output_dir / "mask_and_partition_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_inference": False,
                "new_result_images": False,
                "purity_threshold": args.purity_threshold,
                "pooling": "adaptive_avg_pool2d_occupancy",
                "resolutions": list(RESOLUTIONS),
                "outputs": ["purity_counts.csv", "route_checks.csv", "mask_and_partition_checks.json"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
