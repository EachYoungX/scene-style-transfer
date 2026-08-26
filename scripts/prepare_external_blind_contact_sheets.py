"""Create ordered, method-blind contact sheets for external review."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
BLIND_ROOT = ROOT / "external_benchmark/evaluation/human_blind"
TEMPLATE = BLIND_ROOT / "review_template.csv"
OUTPUT_ROOT = BLIND_ROOT / "contact_sheets"

CASES_PER_SHEET = 12
COLUMNS = 3
TILE_SIZE = 384
PANEL_GAP = 6
LABEL_HEIGHT = 30
TILE_GAP = 18
BACKGROUND = (245, 245, 245)


def font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_panel(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.contain(image.convert("RGB"), (TILE_SIZE, TILE_SIZE))


def make_tile(row: dict[str, str]) -> Image.Image:
    tile_width = TILE_SIZE * 3 + PANEL_GAP * 2
    tile_height = LABEL_HEIGHT + TILE_SIZE
    tile = Image.new("RGB", (tile_width, tile_height), "white")
    draw = ImageDraw.Draw(tile)
    label_font = font(18)
    blind_id = row["blind_id"]
    id_box = draw.textbbox((0, 0), blind_id, font=label_font)
    draw.text((tile_width - (id_box[2] - id_box[0]) - 8, 5), blind_id, fill=(0, 0, 0), font=label_font)
    for index, (name, key) in enumerate(
        (("content", "content_image"), ("reference", "reference_image"), ("output", "output_image"))
    ):
        panel = load_panel(BLIND_ROOT / row[key])
        x = index * (TILE_SIZE + PANEL_GAP)
        y = LABEL_HEIGHT
        tile.paste(panel, (x + (TILE_SIZE - panel.width) // 2, y + (TILE_SIZE - panel.height) // 2))
        draw.rectangle((x, y, x + TILE_SIZE - 1, y + TILE_SIZE - 1), outline=(120, 120, 120), width=1)
        draw.text((x + 6, 5), name, fill=(0, 0, 0), font=label_font)
    return tile


def main() -> None:
    rows = list(csv.DictReader(TEMPLATE.open(newline="", encoding="utf-8")))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    index_rows = []
    sheet_width = COLUMNS * (TILE_SIZE * 3 + PANEL_GAP * 2) + (COLUMNS - 1) * TILE_GAP
    tile_height = LABEL_HEIGHT + TILE_SIZE
    rows_per_column = CASES_PER_SHEET // COLUMNS
    sheet_height = rows_per_column * tile_height + (rows_per_column - 1) * TILE_GAP

    for start in range(0, len(rows), CASES_PER_SHEET):
        batch = rows[start : start + CASES_PER_SHEET]
        sheet_number = start // CASES_PER_SHEET + 1
        sheet = Image.new("RGB", (sheet_width, sheet_height), BACKGROUND)
        first_id = batch[0]["blind_id"]
        last_id = batch[-1]["blind_id"]
        filename = f"blind_sheet_{sheet_number:02d}_{first_id[-4:]}-{last_id[-4:]}.png"
        for slot, row in enumerate(batch):
            column = slot % COLUMNS
            line = slot // COLUMNS
            x = column * ((TILE_SIZE * 3 + PANEL_GAP * 2) + TILE_GAP)
            y = line * (tile_height + TILE_GAP)
            sheet.paste(make_tile(row), (x, y))
            index_rows.append(
                {
                    "blind_id": row["blind_id"],
                    "sheet": f"contact_sheets/{filename}",
                    "slot": str(slot + 1),
                    "grid_position": f"row_{line + 1}_column_{column + 1}",
                }
            )
        sheet.save(OUTPUT_ROOT / filename, optimize=True)

    with (BLIND_ROOT / "contact_sheets_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blind_id", "sheet", "slot", "grid_position"])
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"contact_sheets={len(list(OUTPUT_ROOT.glob('*.png')))} cases={len(rows)}")
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
