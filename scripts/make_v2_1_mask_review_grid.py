"""Create ordered V2.1 review strips: content, optional reference, Subject, Background."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

CASE_TO_STEM = {
    "v1_5_demuth_church": "photo_church",
    "v1_5_kulhanek_snow_winter": "photo_snow_winter",
    "v1_5_demuth_wave": "photo_wave",
}


def labeled_panel(path: Path, label: str, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width = round(image.width * height / image.height)
    panel = image.resize(
        (width, height),
        Image.Resampling.NEAREST if label.startswith("S_") or label.startswith("B_") else Image.Resampling.LANCZOS,
    )
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), label, font=font)
    pad = 8
    draw.rectangle((0, 0, box[2] + pad * 2, box[3] + pad * 2), fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return panel


def join(panels: list[Image.Image], gutter: int = 8) -> Image.Image:
    width = sum(panel.width for panel in panels) + gutter * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), (32, 32, 32))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gutter
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--output-dir", default="data/derived/v2_1_geometry_control/reviews")
    args = parser.parse_args()

    manifest_path = ROOT / "configs/experiment/v2_0_geometry_risk_cases.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = {row["canonical_case_id"]: row for row in csv.DictReader(handle)}
    case_ids = args.case_id or list(CASE_TO_STEM)
    source_root = ROOT / "data/derived/v2_0_geometry_risk"
    mask_root = source_root / "annotations/soft_stylization"
    content_root = source_root / "annotation_sources/content"
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    strips = []
    for case_id in case_ids:
        stem = CASE_TO_STEM[case_id]
        row = rows[case_id]
        panels = [
            labeled_panel(content_root / f"{stem}.png", "content", args.height),
        ]
        if args.include_reference:
            reference = ROOT / row["style_path"]
            if reference.exists():
                panels.append(labeled_panel(reference, "reference", args.height))
        panels.extend(
            [
                labeled_panel(mask_root / f"{stem}_S.png", "S_subject", args.height),
                labeled_panel(mask_root / f"{stem}_B.png", "B_background", args.height),
            ]
        )
        strip = join(panels)
        path = output_dir / f"{stem}_content_reference_S_B.png" if args.include_reference else output_dir / f"{stem}_content_S_B.png"
        strip.save(path)
        strips.append((case_id, strip))
        print(path)

    if strips:
        width = max(image.width for _, image in strips)
        height = sum(image.height for _, image in strips) + 12 * (len(strips) - 1)
        contact = Image.new("RGB", (width, height), (32, 32, 32))
        y = 0
        for _, image in strips:
            contact.paste(image, (0, y))
            y += image.height + 12
        name = "all_content_reference_S_B.png" if args.include_reference else "all_content_S_B.png"
        contact.save(output_dir / name)
        print(output_dir / name)


if __name__ == "__main__":
    main()
