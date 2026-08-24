"""Evaluate Snow seed42 purity probes without generating additional images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESOLUTIONS = (64, 32, 16)
VARIANT_SOURCES = {
    "U": ("runs/ip_adapter_plus_injection/v2_1_regional_pilot", "U"),
    "S_raw": ("runs/ip_adapter_plus_injection/v2_1_regional_pilot", "S_subject"),
    "S_match": ("runs/ip_adapter_plus_injection/v2_1_smatch_pilot", "S_match"),
    "S_sep_neutral": ("runs/ip_adapter_plus_injection/v2_1_purity_probe", "S_sep_neutral"),
    "S_sep_conservative": ("runs/ip_adapter_plus_injection/v2_1_purity_probe", "S_sep_conservative"),
}
CASE_DIR = "v1_5_kulhanek_snow_winter"


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def aggregate_global(records: list[dict], resolution: int) -> dict[str, float | int]:
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
        "records": len(selected),
        "raw_ip_rms": raw_rms,
        "gated_ip_rms": gated_rms,
        "rms_ratio": gated_rms / max(raw_rms, 1e-12),
        "gated_l2_energy": gated_energy,
    }


def aggregate_region(records: list[dict], resolution: int, region: str) -> dict[str, float | int | None]:
    selected = [
        record
        for record in records
        if record.get("spatial_gate_height") == resolution and region in (record.get("region_rms") or {})
    ]
    if not selected:
        return {"token_count": None, "raw_ip_rms": None, "gated_ip_rms": None, "rms_ratio": None}
    raw_energy = gated_energy = raw_count = gated_count = 0.0
    token_count = 0
    for record in selected:
        item = record["region_rms"][region]
        full_tokens = int(record["spatial_gate_height"]) * int(record["spatial_gate_width"])
        raw_elements = float(record["raw_ip_residual_l2"]) ** 2 / max(float(record["raw_ip_residual_rms"]) ** 2, 1e-24)
        gated_elements = float(record["gated_ip_residual_l2"]) ** 2 / max(float(record["gated_ip_residual_rms"]) ** 2, 1e-24)
        raw_hidden = raw_elements / max(full_tokens, 1)
        gated_hidden = gated_elements / max(full_tokens, 1)
        count = int(item["token_count"])
        raw_rms = float(item["raw_ip_rms"] or 0.0)
        gated_rms = float(item["gated_ip_rms"] or 0.0)
        raw_energy += raw_rms**2 * count * raw_hidden
        gated_energy += gated_rms**2 * count * gated_hidden
        raw_count += count * raw_hidden
        gated_count += count * gated_hidden
        token_count = max(token_count, count)
    raw_rms = (raw_energy / max(raw_count, 1e-24)) ** 0.5
    gated_rms = (gated_energy / max(gated_count, 1e-24)) ** 0.5
    return {
        "token_count": token_count,
        "raw_ip_rms": raw_rms,
        "gated_ip_rms": gated_rms,
        "rms_ratio": gated_rms / max(raw_rms, 1e-12),
    }


def rgb_metrics(output: np.ndarray, content: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    delta = np.abs(output - content).mean(axis=2)
    return {f"rgb_mae_content_{name}": float(delta[mask].mean()) for name, mask in masks.items()}


def edge_metrics(output: np.ndarray, content: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    output_edges = cv2.Canny(output.astype(np.uint8), 100, 200) > 0
    content_edges = cv2.Canny(content.astype(np.uint8), 100, 200) > 0
    output_edges &= valid
    content_edges &= valid
    tp = float((output_edges & content_edges).sum())
    precision = tp / max(float(output_edges.sum()), 1.0)
    recall = tp / max(float(content_edges.sum()), 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    distance_to_content = cv2.distanceTransform((~content_edges).astype(np.uint8), cv2.DIST_L2, 3)
    distance_to_output = cv2.distanceTransform((~output_edges).astype(np.uint8), cv2.DIST_L2, 3)
    return {
        "edge_f1_vs_content": f1,
        "edge_chamfer_output_to_content": float(distance_to_content[output_edges].mean()) if output_edges.any() else 0.0,
        "edge_chamfer_content_to_output": float(distance_to_output[content_edges].mean()) if content_edges.any() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/ip_adapter_plus_injection/v2_1_purity_probe/audits")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = ROOT / VARIANT_SOURCES["U"][0] / CASE_DIR / "seed42"
    content = load_rgb(base_dir / "content.png")
    valid = load_mask(base_dir / "masks/valid_eval.png")
    masks = {
        "subject": load_mask(base_dir / "masks/subject.png") & valid,
        "background": load_mask(base_dir / "masks/background.png") & valid,
        "valid": valid,
    }
    residual_rows = []
    rgb_rows = []
    edge_rows = []
    loaded_records: dict[str, list[dict]] = {}
    for label, (root_name, file_label) in VARIANT_SOURCES.items():
        run_dir = ROOT / root_name / CASE_DIR / "seed42"
        records = read_jsonl(run_dir / f"{file_label}_residuals.jsonl")
        loaded_records[label] = records
        image = load_rgb(run_dir / f"{file_label}.png")
        rgb_rows.append({"variant": label, **rgb_metrics(image, content, masks)})
        edge_rows.append({"variant": label, **edge_metrics(image, content, valid)})
        for resolution in RESOLUTIONS:
            values = aggregate_global(records, resolution)
            row = {"variant": label, "resolution": resolution, **values}
            if "region_rms" in records[0]:
                for region in ("pure_subject", "mixed", "pure_background", "valid"):
                    region_values = aggregate_region(records, resolution, region)
                    row[f"{region}_token_count"] = region_values["token_count"]
                    row[f"{region}_raw_ip_rms"] = region_values["raw_ip_rms"]
                    row[f"{region}_gated_ip_rms"] = region_values["gated_ip_rms"]
                    row[f"{region}_rms_ratio"] = region_values["rms_ratio"]
            residual_rows.append(row)

    u_energy = {resolution: next(row["gated_l2_energy"] for row in residual_rows if row["variant"] == "U" and row["resolution"] == resolution) for resolution in RESOLUTIONS}
    for row in residual_rows:
        row["gated_total_ratio_vs_U"] = float(row["gated_l2_energy"]) ** 0.5 / max(float(u_energy[row["resolution"]]) ** 0.5, 1e-12)

    def write_csv(name: str, rows: list[dict]) -> None:
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    write_csv("residual_metrics.csv", residual_rows)
    write_csv("rgb_region_metrics.csv", rgb_rows)
    write_csv("edge_metrics.csv", edge_rows)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "case": "v1_5_kulhanek_snow_winter",
                "seed": 42,
                "new_images": False,
                "variants": list(VARIANT_SOURCES),
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
