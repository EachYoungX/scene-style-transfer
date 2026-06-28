"""Create comparison grids across existing run directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def label_panel(image: Image.Image, label: str, height: int) -> Image.Image:
    width = int(image.width * height / image.height)
    canvas = image.resize((width, height)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    pad = 6
    draw.rectangle((0, 0, bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2), fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--style-sweep-run", default="city_forest_strength_20")
    parser.add_argument("--routed-run", default="city_forest_routed_20")
    parser.add_argument("--routed-method", default="routed_v0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    panels: list[tuple[str, Path]] = []
    sweep_root = root / "runs" / "style_sweep" / args.style_sweep_run / args.case_id
    routed_root = root / "runs" / "routed_v0" / args.routed_run / args.case_id

    panels.append(("content", routed_root / args.routed_method / "content.png"))
    panels.append(("style", routed_root / args.routed_method / "style.png"))
    for sweep_id in ["balanced", "strong_style"]:
        panels.append((sweep_id, sweep_root / sweep_id / "output.png"))
    panels.append((args.routed_method, routed_root / args.routed_method / "output.png"))
    for sweep_id in ["max_style_weak_structure", "style_only_pressure"]:
        panels.append((sweep_id, sweep_root / sweep_id / "output.png"))

    height = 384
    gutter = 8
    images = [(label, label_panel(Image.open(path), label, height)) for label, path in panels if path.exists()]
    grid = Image.new("RGB", (sum(image.width for _, image in images) + gutter * (len(images) - 1), height), (32, 32, 32))
    x = 0
    for _, image in images:
        grid.paste(image, (x, 0))
        x += image.width + gutter

    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
