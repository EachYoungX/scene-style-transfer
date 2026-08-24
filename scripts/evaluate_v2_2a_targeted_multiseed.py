"""Evaluate the targeted V2.2a multi-seed threshold validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluate_v2_2a_safe_strength_frontier import (
    BASE,
    FRONTIER,
    RESOLUTIONS,
    aggregate,
    edge_metrics,
    image,
    mask,
    reference_style_proxy,
    rgb_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "v1_5_demuth_church": (0.2, 0.4, 0.6, 0.8, 1.0),
    "v1_5_kulhanek_snow_winter": (0.2, 0.4, 0.6, 0.8, 1.0),
    "v1_5_demuth_wave": (0.2, 0.4, 0.6, 0.8, 1.0),
}
SEEDS = (42, 123, 777)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def variant_dir(case_id: str, seed: int, multiplier: float) -> tuple[Path, str]:
    frontier_dir = FRONTIER / case_id / f"seed{seed}" / f"lambda_{multiplier:.1f}".replace(".", "p")
    if (frontier_dir / "output.png").exists():
        return frontier_dir, f"lambda_{multiplier:.1f}"
    if multiplier == 1.0:
        return BASE / case_id / f"seed{seed}", "U"
    return frontier_dir, f"lambda_{multiplier:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/ip_adapter_plus_injection/v2_2a_safe_strength_frontier/audits/targeted_multiseed")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    residual_rows: list[dict] = []
    summary_rows: list[dict] = []
    for case_id, multipliers in CASES.items():
        for seed in SEEDS:
            base = BASE / case_id / f"seed{seed}"
            if not (base / "content.png").exists():
                base = BASE / case_id / "seed42"
            content = image(base / "content.png")
            reference = image(base / "style.png")
            valid = mask(base / "masks/valid_eval.png")
            regions = {
                "subject": mask(base / "masks/subject.png") & valid,
                "background": mask(base / "masks/background.png") & valid,
                "global": valid,
            }
            all_multipliers = tuple(dict.fromkeys((*multipliers, 1.0)))
            energy_by_resolution: dict[int, dict[float, float]] = {}
            for multiplier in all_multipliers:
                run_dir, variant = variant_dir(case_id, seed, multiplier)
                is_frontier_output = variant.startswith("lambda_")
                residual_path = run_dir / "residuals.jsonl" if is_frontier_output else run_dir / "U_residuals.jsonl"
                output_path = run_dir / "output.png" if is_frontier_output else run_dir / "U.png"
                records = json.loads("[" + ",".join(line.strip() for line in residual_path.read_text(encoding="utf-8").splitlines() if line.strip()) + "]")
                output = image(output_path)
                for resolution in RESOLUTIONS:
                    values = aggregate(records, resolution)
                    energy_by_resolution.setdefault(resolution, {})[multiplier] = values["gated_l2_energy"]
                    residual_rows.append({"case": case_id, "seed": seed, "lambda": multiplier, "variant": variant, "resolution": resolution, **values})
                summary_rows.append(
                    {
                        "case": case_id,
                        "seed": seed,
                        "lambda": multiplier,
                        "variant": variant,
                        **rgb_metrics(output, content, regions),
                        **edge_metrics(output, content, valid),
                        **reference_style_proxy(output, content, reference),
                        "manual_safe": "",
                        "manual_risk": "",
                        "manual_takeover": "",
                        "manual_style_gain_vs_lambda_0p4": "",
                    }
                )
            for row in residual_rows:
                if row["case"] == case_id and row["seed"] == seed:
                    resolution = int(row["resolution"])
                    u_energy = energy_by_resolution[resolution][1.0]
                    row["residual_ratio_vs_U_same_seed"] = float(row["gated_l2_energy"]) ** 0.5 / max(u_energy**0.5, 1e-12)
            u_total = sum(energy_by_resolution[r][1.0] for r in RESOLUTIONS) if 1.0 in multipliers else None
            for row in residual_rows:
                if row["case"] == case_id and row["seed"] == seed and u_total is not None:
                    total = sum(energy_by_resolution[r][float(row["lambda"])] for r in RESOLUTIONS)
                    row["global_residual_ratio_vs_U_same_seed"] = (total / max(u_total, 1e-24)) ** 0.5
    write_csv(output_dir / "targeted_residual_metrics.csv", residual_rows)
    write_csv(output_dir / "targeted_review_metrics_and_labels.csv", summary_rows)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "cases": list(CASES),
                "seeds": list(SEEDS),
                "threshold_tests": {case: list(values) for case, values in CASES.items()},
                "manual_columns_intentionally_blank": True,
                "u_lambda_1_reused_from_v2_1": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
