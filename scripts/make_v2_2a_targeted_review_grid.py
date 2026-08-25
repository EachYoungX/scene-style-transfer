"""Create review grids for targeted V2.2a multi-seed validation."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/ip_adapter_plus_injection/v2_1_regional_pilot"
FRONTIER = ROOT / "runs/ip_adapter_plus_injection/v2_2a_safe_strength_frontier"
OUTPUT = FRONTIER / "reviews/targeted_multiseed"
SEEDS = (42, 123, 777)
CASES = {
    "v1_5_demuth_church": ("photo_church", (0.2, 0.4, 0.6, 0.8, 1.0)),
    "v1_5_kulhanek_snow_winter": ("photo_snow_winter", (0.2, 0.4, 0.6, 0.8, 1.0)),
    "v1_5_demuth_wave": ("photo_wave", (0.2, 0.4, 0.6, 0.8, 1.0)),
}


def panel(path: Path, label: str, height: int = 300) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im = im.resize((round(im.width * height / im.height), height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((0, 0, box[2] + 16, box[3] + 16), fill=(0, 0, 0))
    draw.text((8, 8), label, fill=(255, 255, 255), font=font)
    return im


def make_case(case_id: str, stem: str, multipliers: tuple[float, ...]) -> Image.Image:
    panels = []
    for seed in SEEDS:
        base = BASE / case_id / f"seed{seed}"
        if not (base / "content.png").exists():
            base = BASE / case_id / "seed42"
        panels.append(panel(base / "content.png", f"seed{seed} content"))
        panels.append(panel(base / "style.png", "reference"))
        for multiplier in multipliers:
            if multiplier == 1.0:
                path = FRONTIER / case_id / f"seed{seed}" / "lambda_1p0" / "output.png"
                if not path.exists():
                    path = BASE / case_id / f"seed{seed}" / "U.png"
                if not path.exists():
                    path = BASE / case_id / "seed42" / "U.png"
                label = f"seed{seed} lambda=1.0 U"
            else:
                path = FRONTIER / case_id / f"seed{seed}" / f"lambda_{multiplier:.1f}".replace(".", "p") / "output.png"
                label = f"seed{seed} lambda={multiplier:.1f}"
            panels.append(panel(path, label))
    gutter = 8
    row_width = sum(item.width for item in panels[: 2 + len(multipliers)]) + gutter * (1 + len(multipliers))
    canvas_height = len(SEEDS) * panels[0].height + (len(SEEDS) - 1) * 12
    canvas = Image.new("RGB", (row_width, canvas_height), (32, 32, 32))
    row_size = 2 + len(multipliers)
    for index, item in enumerate(panels):
        row, column = divmod(index, row_size)
        x = sum(panels[row * row_size + offset].width + gutter for offset in range(column))
        canvas.paste(item, (x, row * (panels[0].height + 12)))
    path = OUTPUT / f"{stem}_targeted_multiseed.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    print(path)
    return canvas


def main() -> None:
    strips = [make_case(case_id, stem, multipliers) for case_id, (stem, multipliers) in CASES.items()]
    canvas = Image.new("RGB", (max(im.width for im in strips), sum(im.height for im in strips) + 12 * (len(strips) - 1)), (32, 32, 32))
    y = 0
    for im in strips:
        canvas.paste(im, (0, y))
        y += im.height + 12
    path = OUTPUT / "all_cases_targeted_multiseed.png"
    canvas.save(path)
    print(path)


if __name__ == "__main__":
    main()
