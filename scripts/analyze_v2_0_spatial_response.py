"""Analyze image-space response after the local rigid gate audit passes."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def save_gray(array: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L").save(path)


def summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "pixels": int(values.size),
        "mean": float(values.mean()) if values.size else 0.0,
        "median": float(np.median(values)) if values.size else 0.0,
        "p95": float(np.percentile(values, 95)) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
    }


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/experiment/v2_0_local_gate_audit.yaml").read_text(encoding="utf-8"))
    exp = config["experiment"]
    paths = config["paths"]
    roi_cfg = config["roi"]
    run_dir = ROOT / exp["run_root"] / exp["case_id"] / f"seed{exp['seed']}"
    uniform_dir = ROOT / paths["uniform_run"]
    out_dir = run_dir / "spatial_response"
    out_dir.mkdir(parents=True, exist_ok=True)

    uniform = load_rgb(uniform_dir / "uniform.png")
    rigid = load_rgb(run_dir / "rigid_suppress_0p00.png")
    rigid_gt = load_mask(ROOT / paths["rigid_mask"]) & load_mask(ROOT / paths["valid_eval_mask"])
    x0, y0, x1, y1 = (roi_cfg[k] for k in ("x0", "y0", "x1", "y1"))
    roi = np.zeros(rigid_gt.shape, dtype=bool)
    roi[y0:y1, x0:x1] = True
    outer = np.zeros(rigid_gt.shape, dtype=bool)
    radius = int(roi_cfg["outer_ring_px"])
    outer[max(0, y0 - radius):min(512, y1 + radius), max(0, x0 - radius):min(512, x1 + radius)] = True
    outer[roi] = False

    abs_rgb = np.abs(uniform.astype(np.int16) - rigid.astype(np.int16)).mean(axis=2)
    save_gray(abs_rgb * 4.0, out_dir / "abs_rgb_diff_x4.png")
    distance = cv2.distanceTransform((~rigid_gt).astype(np.uint8), cv2.DIST_L2, 3)
    edge_uniform = cv2.Canny(cv2.cvtColor(uniform, cv2.COLOR_RGB2GRAY), 100, 200) > 0
    edge_rigid = cv2.Canny(cv2.cvtColor(rigid, cv2.COLOR_RGB2GRAY), 100, 200) > 0
    removed = edge_uniform & ~edge_rigid
    added = edge_rigid & ~edge_uniform
    unchanged = edge_uniform & edge_rigid
    save_gray(removed.astype(np.uint8) * 255, out_dir / "removed_edges.png")
    save_gray(added.astype(np.uint8) * 255, out_dir / "added_edges.png")
    save_gray(unchanged.astype(np.uint8) * 255, out_dir / "unchanged_edges.png")

    regions = {
        "rigid_edge": rigid_gt,
        "center_building_roi": roi,
        "outer_ring": outer,
        "outside_roi": ~roi,
    }
    rgb_by_region = {name: summary(abs_rgb[mask]) for name, mask in regions.items()}
    edge_by_region = {}
    for name, mask in regions.items():
        edge_by_region[name] = {
            "removed": int((removed & mask).sum()),
            "added": int((added & mask).sum()),
            "unchanged": int((unchanged & mask).sum()),
            "uniform_edges": int((edge_uniform & mask).sum()),
            "rigid_edges": int((edge_rigid & mask).sum()),
        }

    distance_bins = [(0, 2), (3, 8), (9, 16), (17, 10_000)]
    distance_response = {}
    for low, high in distance_bins:
        mask = (distance >= low) & (distance <= high)
        distance_response[f"{low}-{high if high < 10000 else 'plus'}px"] = {
            "rgb_diff": summary(abs_rgb[mask]),
            "removed_edges": int((removed & mask).sum()),
            "added_edges": int((added & mask).sum()),
        }

    report = {
        "uniform": str((uniform_dir / "uniform.png").relative_to(ROOT)),
        "rigid": str((run_dir / "rigid_suppress_0p00.png").relative_to(ROOT)),
        "roi": [x0, y0, x1, y1],
        "rgb_diff_by_region": rgb_by_region,
        "edge_diff_by_region": edge_by_region,
        "distance_to_rigid": distance_response,
    }
    (out_dir / "spatial_response.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# V2.0 Spatial Response Analysis",
        "",
        f"- Uniform: `{report['uniform']}`",
        f"- Rigid rho=0: `{report['rigid']}`",
        f"- ROI: `{report['roi']}`",
        "",
        "## RGB difference by region",
        "",
    ]
    for name, values in rgb_by_region.items():
        lines.append(f"- `{name}`: mean `{values['mean']:.4f}`, p95 `{values['p95']:.4f}`, pixels `{values['pixels']}`")
    lines += ["", "## Edge difference by region", ""]
    for name, values in edge_by_region.items():
        lines.append(
            f"- `{name}`: removed `{values['removed']}`, added `{values['added']}`, "
            f"unchanged `{values['unchanged']}`"
        )
    lines += ["", "## Interpretation", "", "This report is downstream of a passed local token-level gate audit."]
    (out_dir / "spatial_response_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out_dir.relative_to(ROOT)), "roi_rgb_mean": rgb_by_region["center_building_roi"]["mean"]}, indent=2))


if __name__ == "__main__":
    main()
