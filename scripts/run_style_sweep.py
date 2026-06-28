"""Run IP-Adapter/Canny style-strength sweeps for selected debug cases."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from run_baseline import RunConfig, run  # noqa: E402
from run_debug_set import read_cases  # noqa: E402


def read_sweep(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def label_panel(image: Image.Image, label: str) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    pad = 6
    draw.rectangle((0, 0, bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2), fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return canvas


def make_sweep_grid(case_dir: Path, sweep_rows: list[dict[str, str]]) -> Path:
    panels: list[tuple[str, Image.Image]] = []
    first_dir = case_dir / sweep_rows[0]["sweep_id"]
    panels.append(("content", Image.open(first_dir / "content.png").convert("RGB")))
    panels.append(("style", Image.open(first_dir / "style.png").convert("RGB")))
    for row in sweep_rows:
        output_path = case_dir / row["sweep_id"] / "output.png"
        panels.append((row["sweep_id"], Image.open(output_path).convert("RGB")))

    size = panels[0][1].size[0]
    gutter = 8
    grid = Image.new("RGB", (len(panels) * size + (len(panels) - 1) * gutter, size), (32, 32, 32))
    x = 0
    for label, image in panels:
        grid.paste(label_panel(image.resize((size, size)), label), (x, 0))
        x += size + gutter
    out_path = case_dir / "style_sweep_grid.png"
    grid.save(out_path)
    return out_path


def build_config(case: dict[str, str], row: dict[str, str], args: argparse.Namespace) -> RunConfig:
    prompt = f"{case['prompt']}, {row['prompt_suffix']}"
    return RunConfig(
        method="ip_adapter_canny",
        content=case["content"],
        style=case["style"],
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=float(row["guidance_scale"]),
        strength=float(row["strength"]),
        controlnet_scale=float(row["controlnet_scale"]),
        ip_adapter_scale=float(row["ip_adapter_scale"]),
        size=args.size,
        model_dir=args.model_dir,
        controlnet_dir=args.controlnet_dir,
        ip_adapter_dir=args.ip_adapter_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/experiment/debug_pairs.csv")
    parser.add_argument("--sweep", default="configs/experiment/style_strength_sweep.csv")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--negative-prompt", default="low quality, blurry, distorted, text, watermark, copied objects")
    parser.add_argument("--model-dir", default="models/sd15")
    parser.add_argument("--controlnet-dir", default="models/controlnet_canny")
    parser.add_argument("--ip-adapter-dir", default="models/ip_adapter")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = project_root / "runs" / "style_sweep" / run_name
    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_root} exists. Use --overwrite or a new --run-name.")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    cases = read_cases(project_root / args.manifest)
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in selected]
    sweep_rows = read_sweep(project_root / args.sweep)

    index: list[dict[str, str]] = []
    for case in cases:
        case_dir = out_root / case["case_id"]
        case_dir.mkdir(parents=True)
        for row in sweep_rows:
            sweep_dir = case_dir / row["sweep_id"]
            print(f"[RUN] {case['case_id']} / {row['sweep_id']}")
            config = build_config(case, row, args)
            run(config, project_root, sweep_dir)
            index.append({"case_id": case["case_id"], "sweep_id": row["sweep_id"], "run_dir": str(sweep_dir.relative_to(project_root))})
        grid_path = make_sweep_grid(case_dir, sweep_rows)
        print(f"[GRID] {grid_path}")

    (out_root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Saved style sweep to {out_root}")


if __name__ == "__main__":
    main()
