"""Run and audit the V2.0 snow seed123 local rigid gate."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def load_binary(path: Path) -> np.ndarray:
    return load_gray(path) > 0


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path)


def save_overlay(image: np.ndarray, mask: np.ndarray, path: Path, color: tuple[int, int, int]) -> None:
    rgb = np.asarray(Image.fromarray(image).convert("RGB"), dtype=np.float32)
    rgb[mask] = rgb[mask] * 0.45 + np.asarray(color, dtype=np.float32) * 0.55
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)


def bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def pool_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.adaptive_max_pool2d(tensor, (height, width))[0, 0].numpy() > 0


def pool_gate(gate: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(gate.astype(np.float32))[None, None]
    return (-F.adaptive_max_pool2d(-tensor, (height, width))[0, 0]).numpy()


def save_gate_overlay(image: np.ndarray, gate: np.ndarray, path: Path) -> None:
    suppressed = np.asarray(
        Image.fromarray((gate < 0.999).astype(np.uint8) * 255, mode="L").resize(
            (image.shape[1], image.shape[0]), Image.Resampling.NEAREST
        )
    ) > 0
    save_overlay(image, suppressed, path, (255, 64, 32))


def run_generation(config: dict, run_dir: Path, roi: tuple[int, int, int, int], outer_ring_px: int) -> None:
    exp = config["experiment"]
    command = [
        sys.executable,
        str(ROOT / "scripts/run_v2_0_rigid_only.py"),
        "--mode",
        "multiseed",
        "--case-id",
        exp["case_id"],
        "--seed",
        str(exp["seed"]),
        "--retain-ratio",
        str(exp["retention"]),
        "--run-root",
        str(Path(exp["run_root"])),
        "--audit-roi",
        ",".join(str(v) for v in roi),
        "--audit-outer-ring-px",
        str(outer_ring_px),
        "--log-residuals",
        "--overwrite",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    if not (run_dir / "rigid_suppress_0p00_residuals.jsonl").exists():
        raise FileNotFoundError("local audit generation completed without residual log")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def grouped_mean(rows: list[dict], key: str) -> list[dict]:
    groups: dict[object, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row.get(key)].append(row)
    output = []
    for group_key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        result = {key: group_key, "calls": len(group)}
        numeric = [
            field
            for field in group[0]
            if field not in {key, "processor_name"}
            and isinstance(group[0].get(field), (int, float))
        ]
        for field in numeric:
            values = [row[field] for row in group if isinstance(row.get(field), (int, float))]
            if values:
                result[field] = float(np.mean(values))
        output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/v2_0_local_gate_audit.yaml")
    parser.add_argument("--no-run", action="store_true", help="reuse an existing audit forward")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    exp = config["experiment"]
    roi_cfg = config["roi"]
    roi = (roi_cfg["x0"], roi_cfg["y0"], roi_cfg["x1"], roi_cfg["y1"])
    run_dir = ROOT / exp["run_root"] / exp["case_id"] / f"seed{exp['seed']}"
    audit_dir = run_dir / "audit_snow_seed123"
    input_dir = audit_dir / "input_alignment"
    gate_dir = audit_dir / "effective_gates"
    local_dir = audit_dir / "local_residual"
    overlay_dir = audit_dir / "overlays"
    residual_path = run_dir / "rigid_suppress_0p00_residuals.jsonl"
    if not args.no_run or not residual_path.exists():
        run_generation(config, run_dir, roi, roi_cfg["outer_ring_px"])
    for directory in (input_dir, gate_dir, local_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source = load_gray(ROOT / config["paths"]["content_source"])
    runtime = load_gray(run_dir / "content.png")
    rigid = load_binary(ROOT / config["paths"]["rigid_mask"])
    valid_content = load_binary(ROOT / config["paths"]["valid_content_mask"])
    valid_eval = load_binary(ROOT / config["paths"]["valid_eval_mask"])
    effective_rigid = rigid & valid_eval
    roi_mask = np.zeros((512, 512), dtype=bool)
    x0, y0, x1, y1 = roi
    roi_mask[y0:y1, x0:x1] = True
    outer_mask = np.zeros((512, 512), dtype=bool)
    radius = int(roi_cfg["outer_ring_px"])
    outer_mask[max(0, y0 - radius):min(512, y1 + radius), max(0, x0 - radius):min(512, x1 + radius)] = True
    outer_mask[roi_mask] = False

    for name, image in (("annotation_source.png", source), ("runtime_content.png", runtime)):
        Image.fromarray(image, mode="L").save(input_dir / name)
    save_mask(rigid, input_dir / "rigid_mask.png")
    save_mask(valid_content, input_dir / "valid_content.png")
    save_mask(valid_eval, input_dir / "valid_eval.png")
    save_overlay(runtime, effective_rigid, input_dir / "rigid_overlay.png", (255, 48, 32))
    save_overlay(runtime, roi_mask, input_dir / "roi_overlay.png", (32, 160, 255))

    alignment = {
        "annotation_source": str(Path(config["paths"]["content_source"])),
        "runtime_content": str((run_dir / "content.png").relative_to(ROOT)),
        "shape_source": list(source.shape),
        "shape_runtime": list(runtime.shape),
        "dtype_source": str(source.dtype),
        "dtype_runtime": str(runtime.dtype),
        "max_abs_pixel_diff": int(np.abs(source.astype(np.int16) - runtime.astype(np.int16)).max()),
        "mean_abs_pixel_diff": float(np.abs(source.astype(np.float32) - runtime.astype(np.float32)).mean()),
        "rigid_pixel_count_total": int(rigid.sum()),
        "rigid_pixel_count_inside_roi": int((rigid & roi_mask).sum()),
        "effective_rigid_pixel_count": int(effective_rigid.sum()),
        "effective_rigid_inside_roi": int((effective_rigid & roi_mask).sum()),
        "removed_by_valid_eval": int((rigid & ~valid_eval).sum()),
        "rigid_bbox": bbox(rigid),
        "roi": list(roi),
        "roi_rigid_bbox": bbox(rigid & roi_mask),
    }
    (input_dir / "alignment_report.json").write_text(json.dumps(alignment, indent=2), encoding="utf-8")

    rows: list[dict] = []
    with residual_path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    resolutions = sorted(
        {(int(row["spatial_gate_height"]), int(row["spatial_gate_width"])) for row in rows}
    )
    source_gate = np.ones((512, 512), dtype=np.float32)
    source_gate[effective_rigid] = float(exp["retention"])
    gate_stats = []
    for height, width in [(512, 512)] + resolutions:
        pooled = pool_gate(source_gate, height, width)
        gate_path = gate_dir / f"gate_{height}x{width}.png"
        Image.fromarray(np.clip(np.rint(pooled * 255), 0, 255).astype(np.uint8), mode="L").save(gate_path)
        upscaled = Image.fromarray(np.clip(np.rint(pooled * 255), 0, 255).astype(np.uint8), mode="L").resize((512, 512), Image.Resampling.NEAREST)
        upscaled.save(gate_dir / f"gate_{height}x{width}_upscaled.png")
        save_gate_overlay(runtime, pooled, overlay_dir / f"gate_{height}x{width}_upscaled_overlay.png")
        rigid_tokens = pool_mask(effective_rigid, height, width)
        gate_stats.append(
            {
                "height": height,
                "width": width,
                "total_tokens": height * width,
                "suppressed_tokens_total": int((pooled < 0.999).sum()),
                "suppressed_fraction_total": float((pooled < 0.999).mean()),
                "rigid_related_tokens": int(rigid_tokens.sum()),
                "rigid_related_suppressed_tokens": int((rigid_tokens & (pooled < 0.999)).sum()),
                "rigid_related_missed_tokens": int((rigid_tokens & (pooled >= 0.999)).sum()),
                "roi_tokens": int(pool_mask(roi_mask, height, width).sum()),
                "roi_suppressed_tokens": int((pool_mask(roi_mask, height, width) & (pooled < 0.999)).sum()),
            }
        )
    (gate_dir / "mapping_report.json").write_text(json.dumps(gate_stats, indent=2), encoding="utf-8")

    per_call_fields = [
        "processor_name", "step", "timestep", "scale", "spatial_gate_height", "spatial_gate_width",
        "rigid_token_count", "roi_token_count", "outer_token_count", "rigid_related_missed_tokens",
        "raw_ip_residual_rms", "gated_ip_residual_rms", "global_rms_ratio",
        "raw_rms_rigid", "gated_rms_rigid", "rigid_rms_ratio",
        "raw_rms_roi", "gated_rms_roi", "roi_rms_ratio",
        "raw_rms_outer", "gated_rms_outer", "outer_rms_ratio",
        "raw_rms_nonrigid", "gated_rms_nonrigid", "nonrigid_rms_ratio",
    ]
    write_csv(local_dir / "per_call.csv", rows, per_call_fields)
    write_csv(local_dir / "per_processor.csv", grouped_mean(rows, "processor_name"), ["processor_name", "calls"] + [f for f in per_call_fields if f not in {"processor_name", "step", "timestep"}])
    write_csv(local_dir / "per_resolution.csv", grouped_mean(rows, "spatial_gate_height"), ["spatial_gate_height", "calls"] + [f for f in per_call_fields if f not in {"processor_name", "step", "timestep", "spatial_gate_height"}])
    write_csv(local_dir / "per_timestep.csv", grouped_mean(rows, "timestep"), ["timestep", "calls"] + [f for f in per_call_fields if f not in {"processor_name", "step", "timestep"}])

    tolerance = float(config["audit"]["raw_gated_ratio_tolerance"])
    identity_tolerance = float(config["audit"]["non_rigid_identity_tolerance"])
    missing = [row for row in rows if int(row.get("rigid_related_missed_tokens") or 0) != 0]
    rigid_ratio_failures = [
        row for row in rows
        if row.get("raw_rms_rigid") and float(row["raw_rms_rigid"]) > 1e-8
        and (row.get("rigid_rms_ratio") is None or float(row["rigid_rms_ratio"]) > tolerance)
    ]
    nonrigid_failures = [
        row for row in rows
        if row.get("raw_rms_nonrigid") and (
            row.get("nonrigid_rms_ratio") is None
            or abs(float(row["nonrigid_rms_ratio"]) - 1.0) > identity_tolerance
        )
    ]
    alignment_pass = alignment["max_abs_pixel_diff"] == 0 and alignment["shape_source"] == alignment["shape_runtime"]
    input_pass = alignment_pass and alignment["rigid_pixel_count_inside_roi"] > 0 and alignment["effective_rigid_inside_roi"] > 0
    mapping_pass = not missing and all(item["rigid_related_missed_tokens"] == 0 for item in gate_stats[1:])
    residual_pass = not rigid_ratio_failures and not nonrigid_failures and all(
        row.get("raw_ip_residual_rms") is not None and row.get("gated_ip_residual_rms") is not None for row in rows
    )
    passed = input_pass and mapping_pass and residual_pass
    summary_lines = [
        "# V2.0 Local Rigid Gate Audit Summary",
        "",
        f"- Case: `{exp['case_id']}`, seed `{exp['seed']}`, rho `{exp['retention']}`",
        f"- ROI: `{roi}`; outer ring: `{radius}px`",
        f"- Active resolutions observed: `{', '.join(f'{h}x{w}' for h, w in resolutions)}`",
        "",
        "## Result",
        "",
        f"**{'LOCAL GATE AUDIT PASSED' if passed else 'LOCAL GATE AUDIT FAILED'}**",
        "",
        f"- Input alignment exact: `{alignment_pass}`; max pixel diff `{alignment['max_abs_pixel_diff']}`",
        f"- Target rigid pixels inside ROI: `{alignment['effective_rigid_inside_roi']}`",
        f"- Token mapping complete: `{mapping_pass}`; missed-call count `{len(missing)}`",
        f"- Same-forward raw→gated residual checks: `{residual_pass}`",
        f"- Rigid ratio failures: `{len(rigid_ratio_failures)}`; non-rigid identity failures: `{len(nonrigid_failures)}`",
        "",
        "## Interpretation",
        "",
        "This audit establishes local token-level implementation correctness only. It does not establish that the final image geometry must remain fixed.",
        "",
        "## Evidence",
        "",
        "- `input_alignment/alignment_report.json`",
        "- `effective_gates/mapping_report.json`",
        "- `local_residual/per_call.csv`",
        "- `overlays/`",
    ]
    (audit_dir / "audit_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps({"audit_dir": str(audit_dir.relative_to(ROOT)), "passed": passed, "resolutions": resolutions}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
