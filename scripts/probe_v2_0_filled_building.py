"""Run the minimal filled-building diagnostic after a passed local gate audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/experiment/v2_0_local_gate_audit.yaml").read_text(encoding="utf-8"))
    exp = config["experiment"]
    paths = config["paths"]
    roi_cfg = config["roi"]
    x0, y0, x1, y1 = (roi_cfg[k] for k in ("x0", "y0", "x1", "y1"))
    expansion = 2
    rigid = load_mask(ROOT / paths["rigid_mask"])
    valid_eval = load_mask(ROOT / paths["valid_eval_mask"])
    filled = rigid.copy()
    filled[max(0, y0 - expansion):min(512, y1 + expansion), max(0, x0 - expansion):min(512, x1 + expansion)] = True
    filled &= valid_eval

    base = ROOT / exp["run_root"] / exp["case_id"] / f"seed{exp['seed']}"
    mask_dir = base / "diagnostic_masks" / "filled_rigid_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(filled.astype(np.uint8) * 255, mode="L").save(mask_dir / "photo_snow_winter.png")
    (base / "diagnostic_masks" / "filled_region_config.json").write_text(
        json.dumps({"roi": [x0, y0, x1, y1], "expansion_px": expansion, "pixels": int(filled.sum())}, indent=2),
        encoding="utf-8",
    )

    filled_root = Path("runs/ip_adapter_plus_injection/v2_0_local_gate_audit_filled")
    command = [
        sys.executable,
        str(ROOT / "scripts/run_v2_0_rigid_only.py"),
        "--mode", "multiseed",
        "--case-id", exp["case_id"],
        "--seed", str(exp["seed"]),
        "--retain-ratio", str(exp["retention"]),
        "--run-root", str(filled_root),
        "--rigid-masks", str(mask_dir.relative_to(ROOT)),
        "--audit-roi", ",".join(str(v) for v in (x0, y0, x1, y1)),
        "--audit-outer-ring-px", str(roi_cfg["outer_ring_px"]),
        "--log-residuals", "--overwrite",
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    uniform = np.asarray(Image.open(ROOT / paths["uniform_run"] / "uniform.png").convert("RGB"), dtype=np.int16)
    edge = np.asarray(Image.open(base / "rigid_suppress_0p00.png").convert("RGB"), dtype=np.int16)
    filled_run = ROOT / filled_root / exp["case_id"] / f"seed{exp['seed']}"
    filled_image = np.asarray(Image.open(filled_run / "rigid_suppress_0p00.png").convert("RGB"), dtype=np.int16)
    roi = np.zeros((512, 512), dtype=bool)
    roi[y0:y1, x0:x1] = True
    edge_diff = np.abs(uniform - edge).mean(axis=2)
    filled_diff = np.abs(uniform - filled_image).mean(axis=2)
    report = {
        "uniform": str((ROOT / paths["uniform_run"] / "uniform.png").relative_to(ROOT)),
        "edge_rigid": str((base / "rigid_suppress_0p00.png").relative_to(ROOT)),
        "filled_rigid": str((filled_run / "rigid_suppress_0p00.png").relative_to(ROOT)),
        "roi": [x0, y0, x1, y1],
        "expansion_px": expansion,
        "roi_rgb_diff_edge_mean": float(edge_diff[roi].mean()),
        "roi_rgb_diff_filled_mean": float(filled_diff[roi].mean()),
        "roi_rgb_diff_edge_p95": float(np.percentile(edge_diff[roi], 95)),
        "roi_rgb_diff_filled_p95": float(np.percentile(filled_diff[roi], 95)),
    }
    out_dir = base / "filled_region_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "filled_region_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "filled_region_summary.md").write_text(
        "\n".join([
            "# V2.0 Filled Building Probe",
            "",
            f"- Edge-only ROI RGB diff mean: `{report['roi_rgb_diff_edge_mean']:.4f}`",
            f"- Filled-region ROI RGB diff mean: `{report['roi_rgb_diff_filled_mean']:.4f}`",
            f"- Edge-only ROI RGB diff p95: `{report['roi_rgb_diff_edge_p95']:.4f}`",
            f"- Filled-region ROI RGB diff p95: `{report['roi_rgb_diff_filled_p95']:.4f}`",
            "",
            "The filled mask is diagnostic only and is not a replacement for the formal rigid GT.",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
