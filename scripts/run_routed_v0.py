"""Run a global-parameter approximation of Routing V0.

This does not perform local attention masking. It maps each case's routing plan
to the global controls exposed by the current SD1.5 + ControlNet + IP-Adapter
baseline, so we can test whether the routing signal points to a better region
of the style/structure trade-off curve.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from run_baseline import RunConfig, run  # noqa: E402


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_plan(routing_dir: Path, case_id: str) -> dict:
    path = routing_dir / case_id / "routing_plan.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def map_plan_to_params(plan: dict, variant: str = "default") -> dict[str, float]:
    local = float(plan["local_style_weight"])
    global_style = float(plan["global_appearance_weight"])
    lock = float(plan["structure_lock_weight"])

    # Default was intentionally conservative; it often lands between balanced
    # and strong_style. The push variant tests the stronger target interval
    # between strong_style and max_style_weak_structure.
    params = {
        "strength": round(clamp(0.52 + 0.28 * global_style - 0.12 * lock + 0.10 * local, 0.52, 0.76), 4),
        "controlnet_scale": round(clamp(0.45 + 0.45 * lock, 0.45, 0.95), 4),
        "ip_adapter_scale": round(clamp(0.55 + 0.55 * global_style + 0.25 * local - 0.20 * lock, 0.55, 1.05), 4),
        "guidance_scale": round(clamp(5.5 + 1.0 * global_style - 0.5 * lock, 5.2, 6.8), 4),
    }
    if variant == "push":
        return {
            "strength": round(clamp(params["strength"] + 0.06, 0.58, 0.80), 4),
            "controlnet_scale": round(clamp(params["controlnet_scale"] - 0.10, 0.55, 0.90), 4),
            "ip_adapter_scale": round(clamp(params["ip_adapter_scale"] + 0.12, 0.70, 1.12), 4),
            "guidance_scale": round(clamp(params["guidance_scale"] - 0.15, 5.0, 6.5), 4),
        }
    if variant != "default":
        raise ValueError(f"Unknown routed V0 variant: {variant}")
    return params


def label_panel(image: Image.Image, label: str) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    pad = 6
    draw.rectangle((0, 0, bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2), fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return canvas


def make_grid(case_dir: Path, method_name: str) -> Path:
    panels = []
    for label, path in [
        ("content", case_dir / method_name / "content.png"),
        ("style", case_dir / method_name / "style.png"),
        (method_name, case_dir / method_name / "output.png"),
        ("risk", case_dir / "risk_preview.png"),
    ]:
        if path.exists():
            panels.append((label, Image.open(path).convert("RGB")))
    size = panels[0][1].height
    gutter = 8
    widths = [int(panel.size[0] * size / panel.size[1]) for _, panel in panels]
    grid = Image.new("RGB", (sum(widths) + gutter * (len(panels) - 1), size), (32, 32, 32))
    x = 0
    for (label, image), width in zip(panels, widths):
        image = image.resize((width, size))
        grid.paste(label_panel(image, label), (x, 0))
        x += width + gutter
    out_path = case_dir / "routed_v0_grid.png"
    grid.save(out_path)
    return out_path


def build_config(case: dict[str, str], params: dict[str, float], args: argparse.Namespace) -> RunConfig:
    prompt = f"{case['prompt']}, controlled strong reference style, preserve scene layout and key geometry"
    return RunConfig(
        method="ip_adapter_canny",
        content=case["content"],
        style=case["style"],
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=params["guidance_scale"],
        strength=params["strength"],
        controlnet_scale=params["controlnet_scale"],
        ip_adapter_scale=params["ip_adapter_scale"],
        size=args.size,
        model_dir=args.model_dir,
        controlnet_dir=args.controlnet_dir,
        ip_adapter_dir=args.ip_adapter_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/experiment/debug_pairs.csv")
    parser.add_argument("--routing-run", default="debug_v0_adjusted")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--variant", choices=["default", "push"], default="default")
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
    routing_dir = project_root / "runs" / "routing_v0" / args.routing_run
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = project_root / "runs" / "routed_v0" / run_name
    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_root} exists. Use --overwrite or a new --run-name.")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    cases = read_cases(project_root / args.manifest)
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in selected]

    index = []
    for case in cases:
        case_dir = out_root / case["case_id"]
        case_dir.mkdir(parents=True)
        plan = load_plan(routing_dir, case["case_id"])
        params = map_plan_to_params(plan, args.variant)
        config = build_config(case, params, args)
        method_name = "routed_v0" if args.variant == "default" else f"routed_v0_{args.variant}"
        method_dir = case_dir / method_name
        print(f"[RUN] {case['case_id']} params={params}")
        run(config, project_root, method_dir)

        risk_preview = routing_dir / case["case_id"] / "risk_preview.png"
        if risk_preview.exists():
            shutil.copy2(risk_preview, case_dir / "risk_preview.png")
        (case_dir / "routing_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        (case_dir / "mapped_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        grid = make_grid(case_dir, method_name)
        print(f"[GRID] {grid}")
        index.append({"case_id": case["case_id"], "variant": args.variant, "params": params, "run_dir": str(case_dir.relative_to(project_root))})

    (out_root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Saved routed V0 run to {out_root}")


if __name__ == "__main__":
    main()
