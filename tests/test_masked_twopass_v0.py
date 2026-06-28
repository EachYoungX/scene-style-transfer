import numpy as np
from PIL import Image

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from run_masked_twopass_v0 import composite  # noqa: E402


def test_composite_uses_safe_in_high_risk_regions():
    strong = Image.new("RGB", (2, 1), (255, 0, 0))
    safe = Image.new("RGB", (2, 1), (0, 0, 255))
    alpha = np.array([[[1.0], [0.0]]], dtype=np.float32)

    result = np.array(composite(strong, safe, alpha))

    assert result[0, 0].tolist() == [0, 0, 255]
    assert result[0, 1].tolist() == [255, 0, 0]
