"""Create seed42 generated-result comparisons including S_match."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "v1_5_demuth_church": "photo_church",
    "v1_5_kulhanek_snow_winter": "photo_snow_winter",
    "v1_5_demuth_wave": "photo_wave",
}
BASE_ROOT = ROOT / "runs/ip_adapter_plus_injection/v2_1_regional_pilot"
MATCH_ROOT = ROOT / "runs/ip_adapter_plus_injection/v2_1_smatch_pilot"
OUTPUT_ROOT = MATCH_ROOT / "reviews"


def panel(path: Path, label: str, height: int = 384) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width = round(image.width * height / image.height)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((0, 0, box[2] + 16, box[3] + 16), fill=(0, 0, 0))
    draw.text((8, 8), label, fill=(255, 255, 255), font=font)
    return image


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
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    strips: list[Image.Image] = []
    for case_id, stem in CASES.items():
        base = BASE_ROOT / case_id / "seed42"
        match = MATCH_ROOT / case_id / "seed42"
        strip = join(
            [
                panel(base / "content.png", "content"),
                panel(base / "style.png", "reference"),
                panel(base / "U.png", "U"),
                panel(base / "S_subject.png", "S_subject"),
                panel(base / "S_background.png", "S_background"),
                panel(match / "S_match.png", "S_match gain=3.5"),
            ]
        )
        path = OUTPUT_ROOT / f"{stem}_seed42_smatch_comparison.png"
        strip.save(path)
        strips.append(strip)
        print(path)

    canvas = Image.new(
        "RGB",
        (max(image.width for image in strips), sum(image.height for image in strips) + 12 * (len(strips) - 1)),
        (32, 32, 32),
    )
    y = 0
    for strip in strips:
        canvas.paste(strip, (0, y))
        y += strip.height + 12
    path = OUTPUT_ROOT / "all_cases_seed42_smatch_comparison.png"
    canvas.save(path)
    print(path)


if __name__ == "__main__":
    main()
