"""Prepare aligned V2.0 annotation sources and editable empty masks."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from annotations.geometry_protocol import (  # noqa: E402
    canny_centerlines,
    edge_difference_helper,
    erode_binary,
    lsd_centerlines,
    require_binary_uint8,
    rigid_centerline_candidate,
)

CONTENT_NAMES = {
    "photo_lecreusois_church": "photo_church.png",
    "photo_sea_wave": "photo_wave.png",
    "photo_snow_winter": "photo_snow_winter.png",
    "photo_seregei_street": "photo_seregei_city.png",
    "photo_seregei_city": "photo_seregei_city.png",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be unique")
    return sorted(rows, key=lambda row: row["sample_id"])


def content_name(row: dict[str, str]) -> str:
    stem = Path(row["content_path"]).stem
    return CONTENT_NAMES.get(stem, f"{stem}.png")


def a2_name(row: dict[str, str]) -> str:
    case_name = row["canonical_case_id"]
    if case_name.startswith("v1_5_"):
        case_name = case_name[len("v1_5_") :]
    return f"{case_name}_seed{int(row['seed'])}.png"


def save_aligned_content(source: Path, destination: Path, size: int) -> None:
    image = Image.open(source).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    canvas.save(destination)


def derive_valid_content_mask(aligned_rgb: np.ndarray) -> np.ndarray:
    non_padding = np.any(aligned_rgb != 0, axis=2)
    rows = np.flatnonzero(non_padding.any(axis=1))
    columns = np.flatnonzero(non_padding.any(axis=0))
    if not rows.size or not columns.size:
        raise ValueError("Aligned content has no non-padding pixels")
    valid = np.zeros(non_padding.shape, dtype=np.uint8)
    valid[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1] = 255
    return valid


def resolve_content_source(rows: list[dict[str, str]]) -> Path:
    raw_source = ROOT / rows[0]["content_path"]
    if raw_source.exists():
        return raw_source
    run_sources = [ROOT / Path(row["output_path"]).parent / "content.png" for row in rows]
    if not all(path.exists() for path in run_sources):
        missing = [str(path) for path in run_sources if not path.exists()]
        raise FileNotFoundError(f"Content source and run-aligned fallback are missing: {missing}")
    reference = np.asarray(Image.open(run_sources[0]).convert("RGB"))
    for path in run_sources[1:]:
        candidate = np.asarray(Image.open(path).convert("RGB"))
        if not np.array_equal(reference, candidate):
            raise ValueError(f"Run-aligned content copies disagree across seeds: {run_sources[0]} and {path}")
    return run_sources[0]


def save_aligned_a2(source: Path, destination: Path, size: int) -> None:
    image = Image.open(source).convert("RGB")
    if image.size != (size, size):
        raise ValueError(
            f"A2 output {source} is {image.size}, expected {(size, size)}. "
            "Regenerate it from frozen run metadata instead of resizing it manually."
        )
    image.save(destination)


def prepare_empty_mask(path: Path, size: int, reset: bool = False) -> None:
    if path.exists() and not reset:
        image = Image.open(path)
        if image.mode != "L" or image.size != (size, size):
            raise ValueError(f"Existing mask must be {size}x{size} 8-bit grayscale: {path}")
        return
    Image.new("L", (size, size), color=0).save(path)


def save_binary_mask(array: np.ndarray, path: Path) -> None:
    require_binary_uint8(array, str(path))
    Image.fromarray(array).save(path)


def is_empty_mask(path: Path) -> bool:
    return not np.asarray(Image.open(path).convert("L"), dtype=np.uint8).any()


def annotation_status(path: Path, previous: str = "") -> str:
    if previous == "complete":
        return "complete"
    return "pending" if is_empty_mask(path) else "in_progress"


def validate_mask_directory(directory: Path, expected_names: set[str], size: int) -> None:
    actual_names = {path.name for path in directory.iterdir() if path.is_file()}
    unexpected = actual_names - expected_names
    if unexpected:
        raise ValueError(f"Unexpected files in mask-only directory {directory}: {sorted(unexpected)}")
    missing = expected_names - actual_names
    if missing:
        raise ValueError(f"Missing masks in {directory}: {sorted(missing)}")
    for name in sorted(expected_names):
        image = Image.open(directory / name)
        if image.mode != "L" or image.size != (size, size):
            raise ValueError(f"Mask must be {size}x{size} 8-bit grayscale: {directory / name}")
        require_binary_uint8(np.asarray(image), str(directory / name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/v2_0_geometry_risk_eval.yaml")
    parser.add_argument("--allow-missing-outputs", action="store_true")
    parser.add_argument(
        "--reset-masks",
        action="store_true",
        help="Replace every existing annotation with a black mask. This destroys manual annotation work.",
    )
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    experiment = config["experiment"]
    output_root = ROOT / experiment["output_root"]
    size = int(experiment["image_size"])
    rows = load_rows(ROOT / experiment["manifest"])
    rigid_config = config["annotations"]["rigid_structure"]
    valid_config = config["annotations"]["valid_masks"]

    manifest_path = ROOT / experiment["annotation_manifest"]
    previous_rows: dict[str, dict[str, str]] = {}
    if manifest_path.exists() and not args.reset_masks:
        previous_rows = {row["sample_id"]: row for row in load_rows(manifest_path)}

    source_root = output_root / "annotation_sources"
    content_root = source_root / "content"
    a2_root = source_root / "a2_outputs"
    helper_root = source_root / "helpers"
    canny_root = helper_root / "canny"
    lsd_root = helper_root / "lsd"
    candidate_root = helper_root / "rigid_candidates"
    edge_difference_root = helper_root / "edge_difference"
    valid_content_root = output_root / "valid_masks" / "valid_content"
    valid_eval_root = output_root / "valid_masks" / "valid_eval"
    annotation_root = output_root / "annotations"
    for directory in (
        content_root,
        a2_root,
        canny_root,
        lsd_root,
        candidate_root,
        edge_difference_root,
        valid_content_root,
        valid_eval_root,
        annotation_root / "rigid_structure",
        annotation_root / "soft_stylization",
        annotation_root / "geometry_failure",
        annotation_root / "uncertainty",
        output_root / "previews",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    content_groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        name = content_name(row)
        content_groups.setdefault(name, []).append(row)

    for name, grouped_rows in sorted(content_groups.items()):
        source = resolve_content_source(grouped_rows)
        save_aligned_content(source, content_root / name, size)
        content_rgb = np.asarray(Image.open(content_root / name).convert("RGB"))
        valid_content = derive_valid_content_mask(content_rgb)
        valid_eval = erode_binary(valid_content, int(valid_config["eval_erosion_radius_px"]))
        save_binary_mask(valid_content, valid_content_root / name)
        save_binary_mask(valid_eval, valid_eval_root / name)
        canny = canny_centerlines(
            content_rgb,
            int(rigid_config["canny_low_threshold"]),
            int(rigid_config["canny_high_threshold"]),
        )
        lsd = lsd_centerlines(content_rgb, float(rigid_config["lsd_min_length_px"]))
        canny = np.where(valid_eval > 0, canny, 0).astype(np.uint8)
        lsd = np.where(valid_eval > 0, lsd, 0).astype(np.uint8)
        candidate = rigid_centerline_candidate(canny, lsd)
        save_binary_mask(canny, canny_root / name)
        save_binary_mask(lsd, lsd_root / name)
        save_binary_mask(candidate, candidate_root / name)
        prepare_empty_mask(annotation_root / "rigid_structure" / name, size, args.reset_masks)
        prepare_empty_mask(annotation_root / "soft_stylization" / name, size, args.reset_masks)

    manifest_rows: list[dict[str, str]] = []
    missing: list[str] = []
    for row in rows:
        sample_id = row["sample_id"]
        previous = previous_rows.get(sample_id, {})
        content_file = content_name(row)
        a2_file = a2_name(row)
        content_source = content_root / content_file
        a2_source = a2_root / a2_file
        source_output = ROOT / row["output_path"]
        if source_output.exists():
            save_aligned_a2(source_output, a2_source, size)
            content_rgb = np.asarray(Image.open(content_source).convert("RGB"))
            output_rgb = np.asarray(Image.open(a2_source).convert("RGB"))
            valid_eval = np.asarray(Image.open(valid_eval_root / content_file), dtype=np.uint8)
            save_binary_mask(
                edge_difference_helper(content_rgb, output_rgb, valid_eval), edge_difference_root / a2_file
            )
            source_status = "ready"
        else:
            source_status = "missing_a2_output"
            missing.append(sample_id)

        rigid_mask = annotation_root / "rigid_structure" / content_file
        soft_mask = annotation_root / "soft_stylization" / content_file
        failure_mask = annotation_root / "geometry_failure" / a2_file
        uncertainty_mask = annotation_root / "uncertainty" / a2_file
        prepare_empty_mask(failure_mask, size, args.reset_masks)
        prepare_empty_mask(uncertainty_mask, size, args.reset_masks)
        manifest_rows.append(
            {
                **row,
                "content_source": str(content_source.relative_to(ROOT)),
                "a2_output_source": str(a2_source.relative_to(ROOT)),
                "risk_path": str((output_root / "risk_maps" / sample_id / "continuous.npy").relative_to(ROOT)),
                "canny_helper": str((canny_root / content_file).relative_to(ROOT)),
                "lsd_helper": str((lsd_root / content_file).relative_to(ROOT)),
                "rigid_candidate": str((candidate_root / content_file).relative_to(ROOT)),
                "edge_difference_helper": str((edge_difference_root / a2_file).relative_to(ROOT)),
                "valid_content_mask": str((valid_content_root / content_file).relative_to(ROOT)),
                "valid_eval_mask": str((valid_eval_root / content_file).relative_to(ROOT)),
                "rigid_structure_mask": str(rigid_mask.relative_to(ROOT)),
                "soft_stylization_mask": str(soft_mask.relative_to(ROOT)),
                "geometry_failure_mask": str(failure_mask.relative_to(ROOT)),
                "uncertainty_mask": str(uncertainty_mask.relative_to(ROOT)),
                "source_status": source_status,
                "rigid_status": annotation_status(rigid_mask, previous.get("rigid_status", "")),
                "soft_status": annotation_status(soft_mask, previous.get("soft_status", "")),
                "geometry_failure_status": annotation_status(
                    failure_mask, previous.get("geometry_failure_status", "")
                ),
                "uncertainty_status": annotation_status(uncertainty_mask, previous.get("uncertainty_status", "")),
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    content_names = set(content_groups)
    a2_names = {a2_name(row) for row in rows}
    validate_mask_directory(annotation_root / "rigid_structure", content_names, size)
    validate_mask_directory(valid_content_root, content_names, size)
    validate_mask_directory(valid_eval_root, content_names, size)
    validate_mask_directory(annotation_root / "soft_stylization", content_names, size)
    validate_mask_directory(annotation_root / "geometry_failure", a2_names, size)
    validate_mask_directory(annotation_root / "uncertainty", a2_names, size)
    if missing and not args.allow_missing_outputs:
        raise FileNotFoundError(f"Missing A2 outputs for: {', '.join(missing)}")
    print(manifest_path)


if __name__ == "__main__":
    main()
