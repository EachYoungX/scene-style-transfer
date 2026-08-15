from pathlib import Path

import cv2
import numpy as np
import pytest
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from annotations.geometry_protocol import (  # noqa: E402
    apply_rigid_priority,
    dilate_binary,
    edge_difference_helper,
    erode_binary,
    require_binary_uint8,
    rigid_centerline_candidate,
)


def test_dilation_is_available_for_metric_tolerance_without_changing_gt():
    ground_truth = np.zeros((11, 11), dtype=np.uint8)
    ground_truth[5, 5] = 255
    tolerant = dilate_binary(ground_truth, radius=2)
    assert ground_truth.sum() == 255
    assert tolerant.sum() > ground_truth.sum()


def test_valid_eval_erodes_content_boundary_by_two_pixels():
    valid_content = np.full((11, 11), 255, dtype=np.uint8)
    valid_eval = erode_binary(valid_content, radius=2)
    assert not valid_eval[0].any()
    assert valid_eval[5, 5] == 255


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


def test_rigid_priority_removes_soft_overlap_and_one_pixel_guard():
    rigid = np.zeros((9, 9), dtype=np.uint8)
    soft = np.full((9, 9), 255, dtype=np.uint8)
    valid = np.full((9, 9), 255, dtype=np.uint8)
    rigid[4, 4] = 255
    rigid_final, soft_final = apply_rigid_priority(rigid, soft, valid, guard_radius=1)
    assert rigid_final[4, 4] == 255
    assert soft_final[4, 4] == 0
    assert soft_final[4, 5] == 0
    assert not ((rigid_final == 255) & (soft_final == 255)).any()
