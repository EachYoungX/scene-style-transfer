"""Create Snow seed42 generated-result review strip for purity probes."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/ip_adapter_plus_injection/v2_1_regional_pilot/v1_5_kulhanek_snow_winter/seed42"
SMATCH = ROOT / "runs/ip_adapter_plus_injection/v2_1_smatch_pilot/v1_5_kulhanek_snow_winter/seed42"
PURITY = ROOT / "runs/ip_adapter_plus_injection/v2_1_purity_probe/v1_5_kulhanek_snow_winter/seed42"
OUTPUT = PURITY / "reviews/snow_seed42_purity_comparison.png"


def panel(path: Path, label: str, height: int = 420) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width = round(image.width * height / image.height)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((0, 0, box[2] + 16, box[3] + 16), fill=(0, 0, 0))
    draw.text((8, 8), label, fill=(255, 255, 255), font=font)
    return image


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        (BASE / "content.png", "content"),
        (BASE / "style.png", "reference"),
        (BASE / "U.png", "U"),
        (BASE / "S_subject.png", "S_raw"),
        (SMATCH / "S_match.png", "S_match"),
        (PURITY / "S_sep_neutral.png", "S_sep_neutral"),
        (PURITY / "S_sep_conservative.png", "S_sep_conservative"),
    ]
    images = [panel(path, label) for path, label in paths]
    gutter = 8
    canvas = Image.new("RGB", (sum(image.width for image in images) + gutter * (len(images) - 1), images[0].height), (32, 32, 32))
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width + gutter
    canvas.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
