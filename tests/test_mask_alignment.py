from pathlib import Path

import numpy as np
import pytest
import sys
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from metrics.mask_utils import (  # noqa: E402
    load_binary_mask,
    load_continuous_risk,
    load_mask,
    validate_alignment,
)


def test_binary_mask_resize_uses_nearest_neighbour(tmp_path):
    source = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    path = tmp_path / "mask.png"
    Image.fromarray(source).save(path)
    resized = load_mask(path, (4, 4))
    assert set(np.unique(resized)) == {0.0, 1.0}
    assert np.all(resized[:2, :2] == 0.0)
    assert np.all(resized[:2, 2:] == 1.0)


def test_continuous_risk_resize_is_bilinear(tmp_path):
    path = tmp_path / "risk.npy"
    np.save(path, np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    resized = load_continuous_risk(path, (5, 5))
    assert 0.0 < resized[2, 2] < 1.0


def test_alignment_rejects_axis_or_size_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_alignment("sample", risk=np.zeros((4, 5)), failure=np.zeros((5, 4)))


def test_alignment_accepts_matching_spatial_shapes():
    shape = validate_alignment(
        "sample", content=np.zeros((4, 5, 3)), risk=np.zeros((4, 5)), failure=np.zeros((4, 5))
    )
    assert shape == (4, 5)


def test_final_binary_mask_accepts_only_absolute_black_and_white(tmp_path):
    path = tmp_path / "mask.png"
    array = np.zeros((512, 512), dtype=np.uint8)
    array[20:30, 40:50] = 255
    Image.fromarray(array).save(path)
    loaded = load_binary_mask(path)
    assert loaded.dtype == bool
    assert loaded.sum() == 100


def test_final_binary_mask_rejects_gray_values(tmp_path):
    path = tmp_path / "mask.png"
    array = np.zeros((512, 512), dtype=np.uint8)
    array[0, 0] = 128
    Image.fromarray(array).save(path)
    with pytest.raises(ValueError, match="only 0 and 255"):
        load_binary_mask(path)
