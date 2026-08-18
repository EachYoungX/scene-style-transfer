"""Probe local rigid gating at one active U-Net spatial resolution at a time."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/experiment/v2_0_local_gate_audit.yaml").read_text(encoding="utf-8"))
    exp = config["experiment"]
    paths = config["paths"]
    roi_cfg = config["roi"]
    roi = np.zeros((512, 512), dtype=bool)
    roi[roi_cfg["y0"]:roi_cfg["y1"], roi_cfg["x0"]:roi_cfg["x1"]] = True
    uniform_path = ROOT / paths["uniform_run"] / "uniform.png"
    edge_path = ROOT / exp["run_root"] / exp["case_id"] / f"seed{exp['seed']}" / "rigid_suppress_0p00.png"
    uniform = image(uniform_path)
    edge = image(edge_path)
    probe_root = Path("runs/ip_adapter_plus_injection/v2_0_local_gate_resolution_probe")
    results = []
    for resolution in (64, 32, 16):
        run_root = probe_root / f"res_{resolution}"
        command = [
            sys.executable,
            str(ROOT / "scripts/run_v2_0_rigid_only.py"),
            "--mode", "multiseed",
            "--case-id", exp["case_id"],
            "--seed", str(exp["seed"]),
            "--retain-ratio", str(exp["retention"]),
            "--run-root", str(run_root),
            "--only-resolution", str(resolution),
            "--audit-roi", ",".join(str(roi_cfg[k]) for k in ("x0", "y0", "x1", "y1")),
            "--audit-outer-ring-px", str(roi_cfg["outer_ring_px"]),
            "--log-residuals", "--overwrite",
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        out_dir = ROOT / run_root / exp["case_id"] / f"seed{exp['seed']}"
        generated = image(out_dir / "rigid_suppress_0p00.png")
        diff_uniform = np.abs(uniform - generated).mean(axis=2)
        diff_edge = np.abs(edge - generated).mean(axis=2)
        residual_path = out_dir / "rigid_suppress_0p00_residuals.jsonl"
        rows = [json.loads(line) for line in residual_path.read_text(encoding="utf-8").splitlines()]
        target = [row for row in rows if int(row["spatial_gate_height"]) == resolution]
        other = [row for row in rows if int(row["spatial_gate_height"]) != resolution]
        results.append(
            {
                "resolution": resolution,
                "run": str(out_dir.relative_to(ROOT)),
                "roi_rgb_diff_vs_uniform_mean": float(diff_uniform[roi].mean()),
                "roi_rgb_diff_vs_edge_mean": float(diff_edge[roi].mean()),
                "global_rgb_diff_vs_uniform_mean": float(diff_uniform.mean()),
                "target_calls": len(target),
                "target_rigid_ratio_mean": float(np.mean([row["rigid_rms_ratio"] for row in target])),
                "other_calls": len(other),
                "other_rigid_ratio_mean": float(np.mean([row["rigid_rms_ratio"] for row in other])),
            }
        )

    out_dir = ROOT / probe_root
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolution_probe.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (out_dir / "resolution_probe.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    lines = [
        "# V2.0 Per-Resolution Local Gate Probe",
        "",
        "Each run gates only one active high-resolution U-Net spatial resolution; all other resolutions are identity.",
        "",
    ]
    for row in results:
        lines.append(
            f"- `{row['resolution']}x{row['resolution']}`: target rigid RMS ratio `{row['target_rigid_ratio_mean']:.6f}`, "
            f"other ratio `{row['other_rigid_ratio_mean']:.6f}`, ROI diff vs Uniform `{row['roi_rgb_diff_vs_uniform_mean']:.4f}`, "
            f"ROI diff vs edge-only `{row['roi_rgb_diff_vs_edge_mean']:.4f}`"
        )
    lines += ["", "Conclusion: this is a causal resolution probe, not a formal geometry score."]
    (out_dir / "resolution_probe_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
