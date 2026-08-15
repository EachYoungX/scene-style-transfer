from pathlib import Path

import numpy as np
import pytest
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from build_v2_0_annotation_manifest import (  # noqa: E402
    a2_name,
    annotation_status,
    content_name,
    prepare_empty_mask,
    save_aligned_a2,
    validate_mask_directory,
)


def row(case_id: str, content_path: str, seed: str = "42") -> dict[str, str]:
    return {
        "canonical_case_id": case_id,
        "content_path": content_path,
        "seed": seed,
    }


def test_annotation_source_names_are_stable():
    assert content_name(row("v1_5_demuth_church", "data/raw/photo_ref/photo_lecreusois_church.jpg")) == "photo_church.png"
    assert content_name(row("v1_5_demuth_wave", "data/raw/photo_ref/photo_sea_wave.jpg")) == "photo_wave.png"
    assert content_name(row("v1_5_kulhanek_snow_winter", "data/raw/photo_ref/photo_snow_winter.jpg")) == "photo_snow_winter.png"
    assert a2_name(row("v1_5_demuth_church", "unused", "123")) == "demuth_church_seed123.png"


def test_empty_masks_are_512_grayscale_and_preserved(tmp_path):
    path = tmp_path / "mask.png"
    prepare_empty_mask(path, 512)
    image = Image.open(path)
    assert image.mode == "L"
    assert image.size == (512, 512)
    assert not np.asarray(image).any()

    painted = np.asarray(image).copy()
    painted[10, 20] = 255
    Image.fromarray(painted, mode="L").save(path)
    prepare_empty_mask(path, 512)
    assert np.asarray(Image.open(path))[10, 20] == 255
    assert annotation_status(path) == "in_progress"
    assert annotation_status(path, "complete") == "complete"


def test_reset_masks_is_explicit(tmp_path):
    path = tmp_path / "mask.png"
    Image.fromarray(np.full((512, 512), 255, dtype=np.uint8), mode="L").save(path)
    prepare_empty_mask(path, 512, reset=True)
    assert not np.asarray(Image.open(path)).any()


def test_a2_output_must_already_match_frozen_size(tmp_path):
    source = tmp_path / "source.png"
    destination = tmp_path / "destination.png"
    Image.new("RGB", (256, 256)).save(source)
    with pytest.raises(ValueError, match="Regenerate"):
        save_aligned_a2(source, destination, 512)


def test_mask_directory_rejects_rgb_source_images(tmp_path):
    Image.new("RGB", (512, 512)).save(tmp_path / "photo_church.png")
    with pytest.raises(ValueError, match="8-bit grayscale"):
        validate_mask_directory(tmp_path, {"photo_church.png"}, 512)
