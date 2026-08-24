"""Create V2.1 generated-result review strips: content, reference, U, S-subject, S-background."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "v1_5_demuth_church": "photo_church",
    "v1_5_kulhanek_snow_winter": "photo_snow_winter",
    "v1_5_demuth_wave": "photo_wave",
}


def panel(path: Path, label: str, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width = round(image.width * height / image.height)
    result = image.resize((width, height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), label, font=font)
    pad = 8
    draw.rectangle((0, 0, box[2] + pad * 2, box[3] + pad * 2), fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return result


def join(images: list[Image.Image], gutter: int = 8) -> Image.Image:
    canvas = Image.new(
        "RGB",
        (sum(image.width for image in images) + gutter * (len(images) - 1), max(image.height for image in images)),
        (32, 32, 32),
    )
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width + gutter
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--output-dir", default="runs/ip_adapter_plus_injection/v2_1_regional_pilot/reviews/results")
    args = parser.parse_args()

    with (ROOT / "configs/experiment/v2_0_geometry_risk_cases.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["canonical_case_id"]: row for row in csv.DictReader(handle)}
    case_ids = args.case_id or list(CASES)
    seeds = args.seed or [42, 123, 777]
    run_root = ROOT / "runs/ip_adapter_plus_injection/v2_1_regional_pilot"
    output_root = ROOT / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    all_strips: list[Image.Image] = []
    for case_id in case_ids:
        stem = CASES[case_id]
        case_strips: list[Image.Image] = []
        for seed in seeds:
            run_dir = run_root / case_id / f"seed{seed}"
            if not (run_dir / "U.png").exists():
                raise FileNotFoundError(run_dir / "U.png")
            images = [
                panel(run_dir / "content.png", f"seed{seed} content", args.height),
                panel(run_dir / "style.png", f"seed{seed} reference", args.height),
                panel(run_dir / "U.png", f"seed{seed} U", args.height),
                panel(run_dir / "S_subject.png", f"seed{seed} S_subject", args.height),
                panel(run_dir / "S_background.png", f"seed{seed} S_background", args.height),
            ]
            strip = join(images)
            strip_path = output_root / f"{stem}_seed{seed}_results.png"
            strip.save(strip_path)
            case_strips.append(strip)
            all_strips.append(strip)
            print(strip_path)

        case_canvas = Image.new(
            "RGB",
            (max(image.width for image in case_strips), sum(image.height for image in case_strips) + 12 * (len(case_strips) - 1)),
            (32, 32, 32),
        )
        y = 0
        for strip in case_strips:
            case_canvas.paste(strip, (0, y))
            y += strip.height + 12
        case_path = output_root / f"{stem}_all_seeds_results.png"
        case_canvas.save(case_path)
        print(case_path)

    if all_strips:
        all_canvas = Image.new(
            "RGB",
            (max(image.width for image in all_strips), sum(image.height for image in all_strips) + 12 * (len(all_strips) - 1)),
            (32, 32, 32),
        )
        y = 0
        for strip in all_strips:
            all_canvas.paste(strip, (0, y))
            y += strip.height + 12
        all_path = output_root / "all_cases_all_seeds_results.png"
        all_canvas.save(all_path)
        print(all_path)


if __name__ == "__main__":
    main()
