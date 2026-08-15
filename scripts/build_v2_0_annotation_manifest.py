"""Build the fixed V2.0 annotation workspace without creating fake masks."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return sorted(csv.DictReader(handle), key=lambda row: row["sample_id"])


def save_square(source: Path, destination: Path, size: int) -> None:
    image = Image.open(source).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    canvas.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/v2_0_geometry_risk_eval.yaml")
    parser.add_argument("--allow-missing-outputs", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    experiment = config["experiment"]
    output_root = ROOT / experiment["output_root"]
    size = int(experiment["image_size"])
    rows = load_rows(ROOT / experiment["manifest"])

    for relative in (
        "content",
        "a2_outputs",
        "annotations/rigid_structure",
        "annotations/geometry_failure",
        "annotations/soft_stylization",
        "annotations/uncertainty",
        "previews",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    missing: list[str] = []
    for row in rows:
        sample_id = row["sample_id"]
        content_out = output_root / "content" / f"{sample_id}.png"
        a2_out = output_root / "a2_outputs" / f"{sample_id}.png"
        save_square(ROOT / row["content_path"], content_out, size)
        source_output = ROOT / row["output_path"]
        if source_output.exists():
            shutil.copy2(source_output, a2_out)
            status = "pending_annotation"
        else:
            status = "missing_a2_output"
            missing.append(sample_id)
        manifest_rows.append(
            {
                **row,
                "content_copy": str(content_out.relative_to(ROOT)),
                "a2_output_copy": str(a2_out.relative_to(ROOT)),
                "risk_path": str((output_root / "risk_maps" / sample_id / "continuous.npy").relative_to(ROOT)),
                "rigid_structure_mask": str((output_root / "annotations/rigid_structure" / f"{sample_id}.png").relative_to(ROOT)),
                "geometry_failure_mask": str((output_root / "annotations/geometry_failure" / f"{sample_id}.png").relative_to(ROOT)),
                "soft_stylization_mask": str((output_root / "annotations/soft_stylization" / f"{sample_id}.png").relative_to(ROOT)),
                "uncertainty_mask": str((output_root / "annotations/uncertainty" / f"{sample_id}.png").relative_to(ROOT)),
                "annotation_status": status,
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    if missing and not args.allow_missing_outputs:
        raise FileNotFoundError(f"Missing A2 outputs for: {', '.join(missing)}")
    print(manifest_path)


if __name__ == "__main__":
    main()
