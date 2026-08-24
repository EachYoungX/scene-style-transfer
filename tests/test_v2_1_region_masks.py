from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from regions.v2_1_masks import load_annotation_binary, load_region_mask_set  # noqa: E402


def _save_rgb(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.repeat(array[:, :, None], 3, axis=2), mode="RGB").save(path)


def _strict(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8) * 255, mode="L").save(path)


def test_rgb_mask_thresholds_and_returns_binary(tmp_path):
    array = np.zeros((512, 512), dtype=np.uint8)
    array[10:20, 10:20] = 200
    path = tmp_path / "subject.png"
    _save_rgb(path, array)
    result = load_annotation_binary(path)
    assert result.dtype == bool
    assert result.sum() == 100


def test_region_masks_exclude_rigid_and_padding(tmp_path):
    subject = np.zeros((512, 512), dtype=np.uint8)
    background = np.full((512, 512), 255, dtype=np.uint8)
    rigid = np.zeros((512, 512), dtype=bool)
    valid_content = np.ones((512, 512), dtype=bool)
    valid_content[:4] = False
    valid_eval = valid_content.copy()
    valid_eval[4:6] = False
    rigid[100:105, 100:105] = True
    subject[100:105, 100:105] = 255
    s_path, b_path = tmp_path / "s.png", tmp_path / "b.png"
    _save_rgb(s_path, subject)
    _save_rgb(b_path, background)
    rigid_path = tmp_path / "rigid.png"
    content_path = tmp_path / "content.png"
    eval_path = tmp_path / "eval.png"
    _strict(rigid_path, rigid)
    _strict(content_path, valid_content)
    _strict(eval_path, valid_eval)

    masks = load_region_mask_set(s_path, b_path, rigid_path, content_path, eval_path)

    assert masks.effective_overlap == 0
    assert masks.subject[102, 102] == 0
    assert masks.background[0, 10] == 0
    assert masks.invalid_excluded_background > 0


def test_unequal_rgb_channels_are_rejected(tmp_path):
    array = np.zeros((512, 512, 3), dtype=np.uint8)
    array[0, 0] = [255, 0, 0]
    path = tmp_path / "bad.png"
    Image.fromarray(array, mode="RGB").save(path)
    with pytest.raises(ValueError, match="unequal channels"):
        load_annotation_binary(path)
