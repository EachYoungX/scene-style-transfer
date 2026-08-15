"""Export frozen R0 geometry-risk maps for the V2.0 sample manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from metrics.geometry_risk_metrics import threshold_risk, top_fraction_risk  # noqa: E402
from preprocess.structure_risk import compute_structure_risk, load_rgb, normalize01  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be unique")
    return sorted(rows, key=lambda row: row["sample_id"])


def save_gray(array: np.ndarray, path: Path) -> None:
    Image.fromarray((np.clip(array, 0.0, 1.0) * 255).round().astype(np.uint8), mode="L").save(path)


def save_binary(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(path)


def heatmap(array: np.ndarray) -> np.ndarray:
    gray = (np.clip(array, 0.0, 1.0) * 255).round().astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def overlay(content: Image.Image, risk: np.ndarray) -> Image.Image:
    base = np.asarray(content, dtype=np.float32)
    colored = heatmap(risk).astype(np.float32)
    alpha = (0.15 + 0.50 * normalize01(risk))[..., None]
    result = base * (1.0 - alpha) + colored * alpha
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/v2_0_geometry_risk_eval.yaml")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    risk_config = config["risk_map"]
    output_root = ROOT / experiment["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    risk_root = output_root / "risk_maps"
    risk_root.mkdir(exist_ok=True)
    rows = read_rows(ROOT / experiment["manifest"])

    source_path = ROOT / risk_config["implementation"]
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    (output_root / "risk_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output_root / "risk_version.json").write_text(
        json.dumps(
            {
                "risk_version": experiment["risk_version"],
                "implementation": risk_config["implementation"],
                "sha256": source_hash,
                "frozen_before_annotation": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_root / "git_commit.json").write_text(
        json.dumps({"commit": git_commit()}, indent=2), encoding="utf-8"
    )

    index: list[dict[str, object]] = []
    for row in rows:
        sample_id = row["sample_id"]
        sample_dir = risk_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        content = load_rgb(ROOT / row["content_path"], int(experiment["image_size"]))
        risk, stats = compute_structure_risk(content, row["scene_type"])
        normalized = normalize01(risk)
        np.save(sample_dir / "continuous.npy", risk.astype(np.float32), allow_pickle=False)
        save_gray(normalized, sample_dir / "normalized.png")
        Image.fromarray(heatmap(normalized)).save(sample_dir / "heatmap.png")
        overlay(content, normalized).save(sample_dir / "overlay.png")

        thresholds: dict[str, float] = {}
        for threshold in risk_config["fixed_thresholds"]:
            label = f"fixed_{float(threshold):.2f}"
            save_binary(threshold_risk(risk, float(threshold)), sample_dir / f"{label}.png")
            thresholds[label] = float(threshold)
        for fraction in risk_config["top_fractions"]:
            mask, cutoff = top_fraction_risk(risk, float(fraction))
            label = f"top_{round(float(fraction) * 100):02d}pct"
            save_binary(mask, sample_dir / f"{label}.png")
            thresholds[label] = cutoff
        record = {"sample_id": sample_id, **stats.__dict__, "thresholds": thresholds}
        index.append(record)
        print(f"[OK] {sample_id}")
    (risk_root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
