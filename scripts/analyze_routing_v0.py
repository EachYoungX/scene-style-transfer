"""Generate V0 structure-risk, coverage, and routing-plan artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from preprocess.structure_risk import compute_structure_risk, load_rgb, save_risk_map
from routing.coverage import estimate_coverage, load_style_manifest
from routing.rule_router import build_routing_plan


def save_risk_preview(content_path: Path, risk_path: Path, out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    content = Image.open(content_path).convert("RGB")
    risk = Image.open(risk_path).convert("RGB")
    gutter = 8
    preview = Image.new("RGB", (content.width * 2 + gutter, content.height), (32, 32, 32))
    preview.paste(content, (0, 0))
    preview.paste(risk, (content.width + gutter, 0))

    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    for label, x in [("content", 0), ("risk", content.width + gutter)]:
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.rectangle((x, 0, x + bbox[2] - bbox[0] + 12, bbox[3] - bbox[1] + 12), fill=(0, 0, 0))
        draw.text((x + 6, 6), label, fill=(255, 255, 255), font=font)

    preview.save(out_path)


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/experiment/debug_pairs.csv")
    parser.add_argument("--style-manifest", default="data/manifests/style_refs.csv")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = project_root / "runs" / "routing_v0" / run_name
    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_root} exists. Use --overwrite or a new --run-name.")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    style_manifest = load_style_manifest(project_root / args.style_manifest)
    summary = []
    for case in read_cases(project_root / args.manifest):
        case_dir = out_root / case["case_id"]
        case_dir.mkdir(parents=True)
        rgb = load_rgb(project_root / case["content"], args.size)
        rgb.save(case_dir / "content.png")

        risk_map, risk_stats = compute_structure_risk(rgb, case["scene_type"])
        risk_path = case_dir / "risk_map.png"
        save_risk_map(risk_map, risk_path)
        save_risk_preview(case_dir / "content.png", risk_path, case_dir / "risk_preview.png")

        coverage = estimate_coverage(case["style"], case["scene_type"], style_manifest)
        plan = build_routing_plan(case["case_id"], coverage, risk_stats)

        (case_dir / "risk_stats.json").write_text(json.dumps(asdict(risk_stats), indent=2), encoding="utf-8")
        (case_dir / "coverage.json").write_text(json.dumps(asdict(coverage), indent=2), encoding="utf-8")
        (case_dir / "routing_plan.json").write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        summary.append(
            {
                "case_id": case["case_id"],
                "coverage": coverage.score,
                "risk_p90": risk_stats.p90,
                "local_style_weight": plan.local_style_weight,
                "global_appearance_weight": plan.global_appearance_weight,
                "structure_lock_weight": plan.structure_lock_weight,
                "regime": plan.recommended_regime,
            }
        )
        print(f"[OK] {case['case_id']} -> {plan.recommended_regime}")

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved V0 routing analysis to {out_root}")


if __name__ == "__main__":
    main()
