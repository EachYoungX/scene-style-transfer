"""Create V2.2a generated-result review strips for seed42."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "v1_5_demuth_church": "photo_church",
    "v1_5_kulhanek_snow_winter": "photo_snow_winter",
    "v1_5_demuth_wave": "photo_wave",
}
BASE = ROOT / "runs/ip_adapter_plus_injection/v2_1_regional_pilot"
FRONTIER = ROOT / "runs/ip_adapter_plus_injection/v2_2a_safe_strength_frontier"
OUTPUT = FRONTIER / "reviews"


def panel(path: Path, label: str, height: int = 320) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width = round(image.width * height / image.height)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((0, 0, box[2] + 16, box[3] + 16), fill=(0, 0, 0))
    draw.text((8, 8), label, fill=(255, 255, 255), font=font)
    return image


def make_case(case_id: str, stem: str) -> Image.Image:
    base = BASE / case_id / "seed42"
    frontier = FRONTIER / case_id / "seed42"
    panels = [
        panel(base / "content.png", "content"),
        panel(base / "style.png", "reference"),
        panel(base / "U.png", "U lambda=1.0"),
    ]
    for value in (0.2, 0.4, 0.6, 0.8):
        panels.append(panel(frontier / f"lambda_{value:.1f}".replace(".", "p") / "output.png", f"lambda={value:.1f}"))
    gutter = 8
    canvas = Image.new("RGB", (sum(item.width for item in panels) + gutter * (len(panels) - 1), panels[0].height), (32, 32, 32))
    x = 0
    for item in panels:
        canvas.paste(item, (x, 0))
        x += item.width + gutter
    path = OUTPUT / f"{stem}_seed42_safe_strength_frontier.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    print(path)
    return canvas


def main() -> None:
    strips = [make_case(case_id, stem) for case_id, stem in CASES.items()]
    canvas = Image.new("RGB", (max(item.width for item in strips), sum(item.height for item in strips) + 12 * (len(strips) - 1)), (32, 32, 32))
    y = 0
    for item in strips:
        canvas.paste(item, (0, y))
        y += item.height + 12
    path = OUTPUT / "all_cases_seed42_safe_strength_frontier.png"
    canvas.save(path)
    print(path)


if __name__ == "__main__":
    main()
