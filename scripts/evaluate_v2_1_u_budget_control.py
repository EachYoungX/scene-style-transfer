"""Evaluate U / U_budget / S_raw for the three seed42 cases."""

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
LABELS = {"U": "U", "U_budget": "U_budget", "S_raw": "S_subject"}
RESOLUTIONS = (64, 32, 16)
BASE = ROOT / "runs/ip_adapter_plus_injection/v2_1_regional_pilot"
UBUDGET = ROOT / "runs/ip_adapter_plus_injection/v2_1_u_budget_control"


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def aggregate(records: list[dict], resolution: int) -> dict[str, float | int]:
    selected = [record for record in records if record.get("spatial_gate_height") == resolution]
    raw_energy = sum(float(record["raw_ip_residual_l2"]) ** 2 for record in selected)
    gated_energy = sum(float(record["gated_ip_residual_l2"]) ** 2 for record in selected)
    raw_count = sum(
        float(record["raw_ip_residual_l2"]) ** 2 / max(float(record["raw_ip_residual_rms"]) ** 2, 1e-24)
        for record in selected
    )
    gated_count = sum(
        float(record["gated_ip_residual_l2"]) ** 2 / max(float(record["gated_ip_residual_rms"]) ** 2, 1e-24)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/ip_adapter_plus_injection/v2_1_u_budget_control/audits")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    residual_rows = []
    rgb_rows = []
    edge_rows = []

    for case_id in CASES:
        base = BASE / case_id / "seed42"
        content = image(base / "content.png")
        valid = mask(base / "masks/valid_eval.png")
        regions = {
            "subject": mask(base / "masks/subject.png") & valid,
            "background": mask(base / "masks/background.png") & valid,
            "global": valid,
        }
        records_by_label = {}
        energy_by_resolution = {}
        for label, file_label in LABELS.items():
            run_dir = UBUDGET / case_id / "seed42" if label == "U_budget" else base
            records = jsonl(run_dir / f"{file_label if label != 'U_budget' else 'U_budget'}_residuals.jsonl")
            records_by_label[label] = records
            output = image(run_dir / f"{file_label if label != 'U_budget' else 'U_budget'}.png")
            rgb_rows.append({"case": case_id, "variant": label, **rgb_metrics(output, content, regions)})
            edge_rows.append({"case": case_id, "variant": label, **edge_metrics(output, content, valid)})
            for resolution in RESOLUTIONS:
                values = aggregate(records, resolution)
                energy_by_resolution.setdefault(resolution, {})[label] = values["gated_l2_energy"]
                residual_rows.append({"case": case_id, "variant": label, "resolution": resolution, **values})
        for row in residual_rows:
            if row["case"] == case_id:
                row["residual_ratio_vs_U"] = float(row["gated_l2_energy"]) ** 0.5 / max(float(energy_by_resolution[row["resolution"]]["U"]) ** 0.5, 1e-12)

    def write_csv(name: str, rows: list[dict]) -> None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    write_csv("residual_metrics.csv", residual_rows)
    write_csv("rgb_region_metrics.csv", rgb_rows)
    write_csv("edge_metrics.csv", edge_rows)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "cases": list(CASES),
                "seed": 42,
                "variants": list(LABELS),
                "resolutions": list(RESOLUTIONS),
                "edge_metrics_are_auxiliary": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
