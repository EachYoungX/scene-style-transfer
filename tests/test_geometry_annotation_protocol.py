from pathlib import Path

import cv2
import numpy as np
import pytest
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from annotations.geometry_protocol import (  # noqa: E402
    dilate_centerline,
    edge_difference_helper,
    require_binary_uint8,
    rigid_centerline_candidate,
)


def test_fixed_radius_produces_nine_pixel_horizontal_band():
    centerline = np.zeros((31, 31), dtype=np.uint8)
    centerline[15, 5:26] = 255
    band = dilate_centerline(centerline, radius=4)
    assert band[:, 15].nonzero()[0].tolist() == list(range(11, 20))
    assert set(np.unique(band)) == {0, 255}


def test_candidate_is_binary_union_of_independent_helpers():
    canny = np.zeros((8, 8), dtype=np.uint8)
    lsd = np.zeros((8, 8), dtype=np.uint8)
    canny[1, 2] = 255
    lsd[5, 6] = 255
    candidate = rigid_centerline_candidate(canny, lsd)
    assert candidate[1, 2] == 255
    assert candidate[5, 6] == 255
    assert candidate.sum() == 510


def test_edge_difference_is_helper_not_soft_score():
    content = np.zeros((32, 32, 3), dtype=np.uint8)
    output = content.copy()
    cv2.line(output, (4, 16), (28, 16), (255, 255, 255), 1)
    difference = edge_difference_helper(content, output)
    assert difference.any()
    assert set(np.unique(difference)) <= {0, 255}


def test_binary_validator_rejects_gray_confidence_values():
    mask = np.array([[0, 128, 255]], dtype=np.uint8)
    with pytest.raises(ValueError, match="only 0 and 255"):
        require_binary_uint8(mask)
