"""Run the frozen debug baseline set and make comparison grids."""

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
from run_baseline import METHODS, RunConfig, run  # noqa: E402


def read_cases(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def draw_label(image: Image.Image, label: str) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), label, font=font)
    width = text_bbox[2] - text_bbox[0]
    height = text_bbox[3] - text_bbox[1]
    pad = 6
    draw.rectangle((0, 0, width + pad * 2, height + pad * 2), fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return canvas


def make_case_grid(case_dir: Path, methods: list[str]) -> Path:
    panels: list[tuple[str, Image.Image]] = []
    content_path = case_dir / methods[0] / "content.png"
    if content_path.exists():
        panels.append(("content", Image.open(content_path).convert("RGB")))

    style_path = case_dir / "ip_adapter_canny" / "style.png"
    if style_path.exists():
        panels.append(("style", Image.open(style_path).convert("RGB")))

    for method in methods:
        output_path = case_dir / method / "output.png"
        if output_path.exists():
            panels.append((method, Image.open(output_path).convert("RGB")))

    if not panels:
        raise FileNotFoundError(f"No output images found under {case_dir}")

    size = panels[0][1].size[0]
    gutter = 8
    grid = Image.new("RGB", (len(panels) * size + (len(panels) - 1) * gutter, size), (32, 32, 32))
    x = 0
    for label, image in panels:
        grid.paste(draw_label(image.resize((size, size)), label), (x, 0))
        x += size + gutter

    out_path = case_dir / "comparison_grid.png"
    grid.save(out_path)
    return out_path


def build_config(case: dict[str, str], method: str, args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        method=method,
        content=case["content"],
        style=case["style"] if method == "ip_adapter_canny" else None,
        prompt=case["prompt"],
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        controlnet_scale=args.controlnet_scale,
        ip_adapter_scale=args.ip_adapter_scale,
        size=args.size,
        model_dir=args.model_dir,
        controlnet_dir=args.controlnet_dir,
        ip_adapter_dir=args.ip_adapter_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/experiment/debug_pairs.csv")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=6.5)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--controlnet-scale", type=float, default=0.8)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.45)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--negative-prompt", default="low quality, blurry, distorted, text, watermark, copied objects")
    parser.add_argument("--model-dir", default="models/sd15")
    parser.add_argument("--controlnet-dir", default="models/controlnet_canny")
    parser.add_argument("--ip-adapter-dir", default="models/ip_adapter")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = project_root / "runs" / "debug_set" / run_name
    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_root} exists. Use --overwrite or a new --run-name.")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    cases = read_cases(project_root / args.manifest)
    index: list[dict[str, str]] = []
    for case in cases:
        case_dir = out_root / case["case_id"]
        case_dir.mkdir(parents=True)
        for method in args.methods:
            config = build_config(case, method, args)
            method_dir = case_dir / method
            print(f"[RUN] {case['case_id']} / {method}")
            run(config, project_root, method_dir)
            index.append({"case_id": case["case_id"], "method": method, "run_dir": str(method_dir.relative_to(project_root))})
        grid_path = make_case_grid(case_dir, args.methods)
        print(f"[GRID] {grid_path}")

    (out_root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Saved debug set to {out_root}")


if __name__ == "__main__":
    main()
