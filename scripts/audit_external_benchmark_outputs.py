"""Audit generated external benchmark outputs without model inference."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUN_MANIFEST = ROOT / "external_benchmark/manifests/runs.csv"
PAIR_MANIFEST = ROOT / "external_benchmark/configs/external_eval_pairs_v1.csv"
OUTPUT_DIR = ROOT / "external_benchmark/evaluation/automatic"


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def resize_like(image: np.ndarray, target: np.ndarray) -> np.ndarray:
    if image.shape[:2] == target.shape[:2]:
        return image
    return cv2.resize(image, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_AREA)


def edge_metrics(output: np.ndarray, content: np.ndarray) -> tuple[float, float]:
    out_edges = cv2.Canny(cv2.cvtColor(output, cv2.COLOR_RGB2GRAY), 100, 200) > 0
    content_edges = cv2.Canny(cv2.cvtColor(content, cv2.COLOR_RGB2GRAY), 100, 200) > 0
    if not out_edges.any() and not content_edges.any():
        return 1.0, 0.0
    if not out_edges.any() or not content_edges.any():
        return 0.0, 1.0
    kernel = np.ones((3, 3), dtype=np.uint8)
    out_dilated = cv2.dilate(out_edges.astype(np.uint8), kernel) > 0
    content_dilated = cv2.dilate(content_edges.astype(np.uint8), kernel) > 0
    precision = float((out_edges & content_dilated).sum() / max(out_edges.sum(), 1))
    recall = float((content_edges & out_dilated).sum() / max(content_edges.sum(), 1))
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    out_distance = cv2.distanceTransform((~content_edges).astype(np.uint8), cv2.DIST_L2, 3)
    content_distance = cv2.distanceTransform((~out_edges).astype(np.uint8), cv2.DIST_L2, 3)
    chamfer = float(
        (out_distance[out_edges].mean() + content_distance[content_edges].mean()) / 2.0
    )
    return f1, chamfer


def color_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_stats = np.concatenate([left.mean(axis=(0, 1)), left.std(axis=(0, 1))]).astype(np.float64)
    right_stats = np.concatenate([right.mean(axis=(0, 1)), right.std(axis=(0, 1))]).astype(np.float64)
    denominator = np.linalg.norm(left_stats) * np.linalg.norm(right_stats)
    return float(np.dot(left_stats, right_stats) / denominator) if denominator else 0.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with PAIR_MANIFEST.open(newline="", encoding="utf-8") as handle:
        pairs = {row["pair_id"]: row for row in csv.DictReader(handle)}
    with RUN_MANIFEST.open(newline="", encoding="utf-8") as handle:
        runs = list(csv.DictReader(handle))

    records: list[dict[str, object]] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    for run in runs:
        if run["status"] != "success":
            continue
        pair = pairs[run["pair_id"]]
        output_path = ROOT / run["output_path"]
        content_path = ROOT / pair["content_path"]
        reference_path = ROOT / pair["reference_path"]
        record: dict[str, object] = {
            "track_id": run["track_id"],
            "method": run["method"],
            "pair_id": run["pair_id"],
            "seed": run["seed"],
            "output_path": run["output_path"],
            "exists": output_path.exists(),
            "readable": False,
            "shape_ok": False,
            "rgb_ok": False,
            "non_degenerate": False,
            "sha256": "",
            "edge_f1": "",
            "edge_chamfer_px": "",
            "reference_color_cosine": "",
            "content_color_cosine": "",
            "audit_status": "missing",
        }
        if output_path.exists():
            try:
                output = load_rgb(output_path)
                content = resize_like(load_rgb(content_path), output)
                reference = resize_like(load_rgb(reference_path), output)
                digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
                f1, chamfer = edge_metrics(output, content)
                record.update(
                    {
                        "readable": True,
                        "shape_ok": output.shape[:2] == (512, 512),
                        "rgb_ok": output.ndim == 3 and output.shape[2] == 3,
                        "non_degenerate": float(output.std()) >= 2.0,
                        "sha256": digest,
                        "edge_f1": round(f1, 6),
                        "edge_chamfer_px": round(chamfer, 6),
                        "reference_color_cosine": round(color_cosine(output, reference), 6),
                        "content_color_cosine": round(color_cosine(output, content), 6),
                    }
                )
                record["audit_status"] = (
                    "pass"
                    if record["shape_ok"] and record["rgb_ok"] and record["non_degenerate"]
                    else "check_format_or_degeneracy"
                )
                hashes[digest].append(f"{run['method']}|{run['pair_id']}|{run['seed']}")
            except Exception as exc:  # keep the audit complete when one file is corrupt
                record["audit_status"] = f"unreadable_{type(exc).__name__}"
        records.append(record)

    fieldnames = list(records[0]) if records else []
    with (OUTPUT_DIR / "output_audit_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    duplicate_groups = [members for members in hashes.values() if len(members) > 1]
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["track_id"]), str(record["method"]))].append(record)
    summary = []
    for (track_id, method), group in sorted(grouped.items()):
        valid = [row for row in group if row["audit_status"] == "pass"]
        summary.append(
            {
                "track_id": track_id,
                "method": method,
                "success_rows_audited": len(group),
                "format_pass_rows": len(valid),
                "mean_edge_f1": round(float(np.mean([row["edge_f1"] for row in valid])), 6) if valid else "",
                "mean_edge_chamfer_px": round(float(np.mean([row["edge_chamfer_px"] for row in valid])), 6) if valid else "",
                "mean_reference_color_cosine": round(float(np.mean([row["reference_color_cosine"] for row in valid])), 6) if valid else "",
                "mean_content_color_cosine": round(float(np.mean([row["content_color_cosine"] for row in valid])), 6) if valid else "",
            }
        )
    with (OUTPUT_DIR / "output_audit_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]) if summary else [])
        writer.writeheader()
        writer.writerows(summary)

    report = {
        "success_manifest_rows": len(records),
        "format_pass_rows": sum(row["audit_status"] == "pass" for row in records),
        "duplicate_sha256_groups": duplicate_groups,
        "failed_manifest_rows_excluded": sum(row["status"] != "success" for row in runs),
        "interpretation": "Auxiliary consistency and proxy metrics only; human absolute scoring remains primary.",
    }
    (OUTPUT_DIR / "output_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
