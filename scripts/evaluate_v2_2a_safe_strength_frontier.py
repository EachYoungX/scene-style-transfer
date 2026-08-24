"""Evaluate the V2.2a global safe-strength frontier against existing U."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CASES = ("v1_5_demuth_church", "v1_5_kulhanek_snow_winter", "v1_5_demuth_wave")
LAMBDAS = (0.2, 0.4, 0.6, 0.8, 1.0)
RESOLUTIONS = (64, 32, 16)
BASE = ROOT / "runs/ip_adapter_plus_injection/v2_1_regional_pilot"
FRONTIER = ROOT / "runs/ip_adapter_plus_injection/v2_2a_safe_strength_frontier"


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def label(value: float) -> str:
    return "U" if value == 1.0 else f"lambda_{value:.1f}".replace(".", "p")


def aggregate(records: list[dict], resolution: int) -> dict[str, float]:
    selected = [record for record in records if record.get("spatial_gate_height") == resolution]
    raw_energy = sum(float(record["raw_ip_residual_l2"]) ** 2 for record in selected)
    gated_energy = sum(float(record["gated_ip_residual_l2"]) ** 2 for record in selected)
    raw_count = sum(
        float(record["raw_ip_residual_l2"]) ** 2
        / max(float(record["raw_ip_residual_rms"]) ** 2, 1e-24)
        for record in selected
    )
    gated_count = sum(
        float(record["gated_ip_residual_l2"]) ** 2
        / max(float(record["gated_ip_residual_rms"]) ** 2, 1e-24)
        for record in selected
    )
    raw_rms = (raw_energy / max(raw_count, 1e-24)) ** 0.5
    gated_rms = (gated_energy / max(gated_count, 1e-24)) ** 0.5
    return {
        "raw_ip_rms": raw_rms,
        "gated_ip_rms": gated_rms,
        "rms_ratio": gated_rms / max(raw_rms, 1e-12),
        "gated_l2_energy": gated_energy,
    }


def rgb_metrics(output: np.ndarray, content: np.ndarray, regions: dict[str, np.ndarray]) -> dict[str, float]:
    delta = np.abs(output - content).mean(axis=2)
    return {f"rgb_mae_output_content_{name}": float(delta[region].mean()) for name, region in regions.items()}


def edge_metrics(output: np.ndarray, content: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    output_edge = cv2.Canny(output.astype(np.uint8), 100, 200) > 0
    content_edge = cv2.Canny(content.astype(np.uint8), 100, 200) > 0
    output_edge &= valid
    content_edge &= valid
    tp = float((output_edge & content_edge).sum())
    precision = tp / max(float(output_edge.sum()), 1.0)
    recall = tp / max(float(content_edge.sum()), 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    output_to_content = cv2.distanceTransform((~content_edge).astype(np.uint8), cv2.DIST_L2, 3)
    content_to_output = cv2.distanceTransform((~output_edge).astype(np.uint8), cv2.DIST_L2, 3)
    return {
        "edge_f1_vs_content": f1,
        "edge_chamfer_output_to_content": float(output_to_content[output_edge].mean()) if output_edge.any() else 0.0,
        "edge_chamfer_content_to_output": float(content_to_output[content_edge].mean()) if content_edge.any() else 0.0,
    }


def reference_style_proxy(output: np.ndarray, content: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Color/statistic proximity to reference; auxiliary, not a style metric."""
    out_mean, ref_mean = output.mean(axis=(0, 1)), reference.mean(axis=(0, 1))
    out_std, ref_std = output.std(axis=(0, 1)), reference.std(axis=(0, 1))
    content_mean, content_std = content.mean(axis=(0, 1)), content.std(axis=(0, 1))
    mean_ref = float(np.linalg.norm(out_mean - ref_mean))
    mean_content = float(np.linalg.norm(content_mean - ref_mean))
    std_ref = float(np.linalg.norm(out_std - ref_std))
    std_content = float(np.linalg.norm(content_std - ref_std))
    reference_color_similarity = 1.0 - mean_ref / max(mean_content, 1e-6)
    reference_contrast_similarity = 1.0 - std_ref / max(std_content, 1e-6)
    return {
        "reference_mean_distance": mean_ref,
        "content_reference_mean_distance": mean_content,
        "reference_color_similarity_vs_content": reference_color_similarity,
        "reference_std_distance": std_ref,
        "content_reference_std_distance": std_content,
        "reference_contrast_similarity_vs_content": reference_contrast_similarity,
        "output_content_global_mae": float(np.abs(output - content).mean()),
        "output_reference_global_mae": float(np.abs(output - reference).mean()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/ip_adapter_plus_injection/v2_2a_safe_strength_frontier/audits")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    residual_rows: list[dict] = []
    rgb_rows: list[dict] = []
    edge_rows: list[dict] = []
    style_rows: list[dict] = []

    for case_id in CASES:
        base = BASE / case_id / "seed42"
        content = image(base / "content.png")
        reference = image(base / "style.png")
        valid = mask(base / "masks/valid_eval.png")
        regions = {
            "subject": mask(base / "masks/subject.png") & valid,
            "background": mask(base / "masks/background.png") & valid,
            "global": valid,
        }
        records_by_lambda: dict[float, list[dict]] = {}
        energy_by_resolution: dict[int, dict[float, float]] = {}
        for multiplier in LAMBDAS:
            variant = label(multiplier)
            if multiplier == 1.0:
                run_dir = base
                residual_path = run_dir / "U_residuals.jsonl"
                output_path = run_dir / "U.png"
            else:
                run_dir = FRONTIER / case_id / "seed42" / variant
                residual_path = run_dir / "residuals.jsonl"
                output_path = run_dir / "output.png"
            records = jsonl(residual_path)
            output = image(output_path)
            records_by_lambda[multiplier] = records
            rgb_rows.append({"case": case_id, "lambda": multiplier, "variant": variant, **rgb_metrics(output, content, regions)})
            edge_rows.append({"case": case_id, "lambda": multiplier, "variant": variant, **edge_metrics(output, content, valid)})
            style_rows.append({"case": case_id, "lambda": multiplier, "variant": variant, **reference_style_proxy(output, content, reference)})
            for resolution in RESOLUTIONS:
                values = aggregate(records, resolution)
                energy_by_resolution.setdefault(resolution, {})[multiplier] = values["gated_l2_energy"]
                residual_rows.append({"case": case_id, "lambda": multiplier, "variant": variant, "resolution": resolution, **values})
        for row in residual_rows:
            if row["case"] == case_id:
                resolution = int(row["resolution"])
                row["residual_ratio_vs_U"] = float(row["gated_l2_energy"]) ** 0.5 / max(energy_by_resolution[resolution][1.0] ** 0.5, 1e-12)
        total_u = sum(energy_by_resolution[resolution][1.0] for resolution in RESOLUTIONS)
        for multiplier in LAMBDAS:
            total = sum(energy_by_resolution[resolution][multiplier] for resolution in RESOLUTIONS)
            matching = [row for row in residual_rows if row["case"] == case_id and row["lambda"] == multiplier]
            for row in matching:
                row["global_residual_ratio_vs_U"] = (total / max(total_u, 1e-24)) ** 0.5

    write_csv(output_dir / "residual_metrics.csv", residual_rows)
    write_csv(output_dir / "rgb_structure_metrics.csv", rgb_rows)
    write_csv(output_dir / "edge_metrics.csv", edge_rows)
    write_csv(output_dir / "reference_style_proxy_metrics.csv", style_rows)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "cases": list(CASES),
                "seed": 42,
                "lambdas": list(LAMBDAS),
                "resolutions": list(RESOLUTIONS),
                "existing_lambda_1_reused": True,
                "edge_metrics_are_auxiliary": True,
                "reference_style_proxy_is_auxiliary": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
