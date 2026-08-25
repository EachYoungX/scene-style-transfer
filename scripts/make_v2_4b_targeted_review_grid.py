"""Create V2.4b seed42 review strips for the targeted pair expansion."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
LAMBDAS = (0.2, 0.4, 0.6, 0.8, 1.0)


def panel(path: Path, label: str, height: int = 300) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = image.resize(
        (round(image.width * height / image.height), height), Image.Resampling.LANCZOS
    )
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((0, 0, box[2] + 16, box[3] + 16), fill=(0, 0, 0))
    draw.text((8, 8), label, fill=(255, 255, 255), font=font)
    return image


def lambda_dir(value: float) -> str:
    return f"lambda_{value:.1f}".replace(".", "p")


def make_case(row: dict[str, str], output_root: Path, output_dir: Path) -> Image.Image:
    case_id = row["case_id"]
    case_root = output_root / case_id / f"seed{row['seed']}"
    panels = [
        panel(ROOT / row["content_path"], f"{case_id} content"),
        panel(ROOT / row["style_path"], "reference"),
    ]
    for value in LAMBDAS:
        panels.append(panel(case_root / lambda_dir(value) / "output.png", f"lambda={value:.1f}"))

    gutter = 8
    width = sum(item.width for item in panels) + gutter * (len(panels) - 1)
    canvas = Image.new("RGB", (width, panels[0].height), (32, 32, 32))
    x = 0
    for item in panels:
        canvas.paste(item, (x, 0))
        x += item.width + gutter
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{case_id}_seed{row['seed']}.png"
    canvas.save(path)
    print(path)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/experiment/v2_4b_targeted_profile_candidates.csv")
    parser.add_argument(
        "--output-root",
        default="runs/ip_adapter_plus_injection/v2_4b_targeted_profile_candidates",
    )
    parser.add_argument(
        "--review-dir",
        default="runs/ip_adapter_plus_injection/v2_4b_targeted_profile_candidates/reviews",
    )
    args = parser.parse_args()

    manifest = ROOT / args.manifest
    output_root = ROOT / args.output_root
    review_dir = ROOT / args.review_dir
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    strips = [make_case(row, output_root, review_dir / "cases") for row in rows]
    gutter = 12
    canvas = Image.new(
        "RGB",
        (max(item.width for item in strips), sum(item.height for item in strips) + gutter * (len(strips) - 1)),
        (32, 32, 32),
    )
    y = 0
    for strip in strips:
        canvas.paste(strip, (0, y))
        y += strip.height + gutter
    path = review_dir / "all_cases_targeted_seed42.png"
    review_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    print(path)


if __name__ == "__main__":
    main()
