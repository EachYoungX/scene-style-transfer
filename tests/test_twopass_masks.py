import numpy as np

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from run_masked_twopass_v0 import oracle_corridor_mask  # noqa: E402


def test_oracle_forest_mask_has_substantial_coverage():
    alpha = oracle_corridor_mask((128, 128), "debug_forest", blur=2.0)

    assert alpha.shape == (128, 128, 1)
    assert float((alpha > 0.2).mean()) > 0.25
