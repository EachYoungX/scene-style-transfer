"""Create V2.0 annotation previews and post-annotation five-panel figures."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from metrics.mask_utils import load_binary_mask, load_continuous_risk, load_rgb  # noqa: E402


def labeled_panel(array: np.ndarray, label: str, size: int = 384) -> Image.Image:
    panel = Image.fromarray(array.astype(np.uint8)).resize((size, size), Image.Resampling.LANCZOS).convert("RGB")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((0, 0, box[2] + 12, box[3] + 12), fill=(0, 0, 0))
    draw.text((6, 6), label, fill=(255, 255, 255), font=font)
    return panel


def risk_heatmap(risk: np.ndarray) -> np.ndarray:
    gray = (np.clip(risk, 0.0, 1.0) * 255).round().astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def mask_rgb(mask: np.ndarray) -> np.ndarray:
    gray = np.asarray(mask * 255, dtype=np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def comparison_overlay(risk: np.ndarray, failure: np.ndarray, soft: np.ndarray, cutoff: float) -> np.ndarray:
    high = risk >= cutoff
    canvas = np.full((*risk.shape, 3), 28, dtype=np.uint8)
    canvas[soft] = (0, 150, 70)  # green: soft region
    canvas[high & ~failure] = (255, 210, 0)  # yellow: predicted only
    canvas[~high & failure] = (30, 110, 255)  # blue: missed failure
    canvas[high & failure] = (235, 45, 45)  # red: covered failure
    return canvas


def join_panels(panels: list[Image.Image], gutter: int = 8) -> Image.Image:
    width = sum(panel.width for panel in panels) + gutter * (len(panels) - 1)
    canvas = Image.new("RGB", (width, max(panel.height for panel in panels)), (20, 20, 20))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gutter
    return canvas


def save_contact_sheet(images: list[Image.Image], path: Path) -> None:
    thumbnails = [image.resize((image.width // 2, image.height // 2), Image.Resampling.LANCZOS) for image in images]
    width = max(image.width for image in thumbnails)
    height = sum(image.height for image in thumbnails) + 8 * (len(thumbnails) - 1)
    contact = Image.new("RGB", (width, height), (20, 20, 20))
    y = 0
    for image in thumbnails:
        contact.paste(image, (0, y))
        y += image.height + 8
    contact.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/v2_0_geometry_risk_eval.yaml")
    parser.add_argument("--overlay-threshold", type=float, default=0.50)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    output_root = ROOT / config["experiment"]["output_root"]
    manifest_path = ROOT / config["experiment"]["annotation_manifest"]
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["sample_id"])
    preview_root = output_root / "previews"
    preview_root.mkdir(parents=True, exist_ok=True)

    previews: list[Image.Image] = []
    blind_previews: list[Image.Image] = []
    for row in rows:
        sample_id = row["sample_id"]
        content = load_rgb(ROOT / row["content_source"])
        output = load_rgb(ROOT / row["a2_output_source"])
        risk = load_continuous_risk(ROOT / row["risk_path"])
        blind = join_panels(
            [
                labeled_panel(content, "content"),
                labeled_panel(output, "A2 output"),
            ]
        )
        blind.save(preview_root / f"{sample_id}_blind.png")
        blind_previews.append(blind)
        panels = [
            labeled_panel(content, "content"),
            labeled_panel(output, "A2 output"),
            labeled_panel(risk_heatmap(risk), "continuous risk"),
        ]
        failure_path = ROOT / row["geometry_failure_mask"]
        soft_path = ROOT / row["soft_stylization_mask"]
        annotations_complete = all(
            row.get(field) == "complete"
            for field in ("rigid_status", "soft_status", "geometry_failure_status", "uncertainty_status")
        )
        if annotations_complete and failure_path.exists() and soft_path.exists():
            failure = load_binary_mask(failure_path)
            soft = load_binary_mask(soft_path)
            panels.extend(
                [
                    labeled_panel(mask_rgb(failure), "failure mask"),
                    labeled_panel(comparison_overlay(risk, failure, soft, args.overlay_threshold), "risk/failure overlay"),
                ]
            )
        preview = join_panels(panels)
        preview.save(preview_root / f"{sample_id}.png")
        previews.append(preview)
        print(f"[OK] {sample_id}")

    save_contact_sheet(blind_previews, preview_root / "all_samples_blind_annotation_sheet.png")
    save_contact_sheet(previews, preview_root / "all_samples_diagnostic_sheet.png")


if __name__ == "__main__":
    main()
