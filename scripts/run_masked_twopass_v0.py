"""Spatially composite safe and strong stylization candidates using a risk map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def risk_mask_from_heatmap(path: Path, threshold: float, softness: float) -> np.ndarray:
    """Return high-risk alpha in [0, 1] from an inferno heatmap image."""
    heatmap = np.array(load_rgb(path)).astype(np.float32) / 255.0
    # Inferno high-risk colors are bright and red/yellow; this proxy is stable
    # enough for generated V0 risk maps and keeps the script dependency-light.
    score = 0.55 * heatmap[..., 0] + 0.30 * heatmap[..., 1] - 0.15 * heatmap[..., 2]
    score = np.clip(score, 0, 1)
    alpha = np.clip((score - threshold) / max(1e-6, softness), 0, 1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=8.0)
    return np.clip(alpha[..., None], 0, 1)


def hard_dilated_risk_mask(path: Path, threshold: float, dilation: int, close: int, blur: float) -> np.ndarray:
    soft = risk_mask_from_heatmap(path, threshold=threshold, softness=0.08)[..., 0]
    mask = (soft > 0.1).astype(np.uint8)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=blur)
    return np.clip(alpha[..., None], 0, 1)


def oracle_corridor_mask(size: tuple[int, int], case_id: str, blur: float) -> np.ndarray:
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    if case_id == "debug_forest":
        # Protect the lower path, path-side boundaries, and central vanishing corridor.
        pts = np.array(
            [
                (int(0.28 * w), h),
                (int(0.43 * w), int(0.46 * h)),
                (int(0.57 * w), int(0.46 * h)),
                (int(0.72 * w), h),
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts], 1)
        cv2.rectangle(mask, (int(0.18 * w), 0), (int(0.30 * w), h), 1, thickness=-1)
        cv2.rectangle(mask, (int(0.70 * w), 0), (int(0.84 * w), h), 1, thickness=-1)
    elif case_id == "debug_city_architecture":
        cv2.rectangle(mask, (int(0.12 * w), int(0.08 * h)), (int(0.88 * w), int(0.88 * h)), 1, thickness=-1)
    else:
        cv2.rectangle(mask, (0, int(0.38 * h)), (w, int(0.62 * h)), 1, thickness=-1)
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=blur)
    return np.clip(alpha[..., None], 0, 1)


def composite(strong: Image.Image, safe: Image.Image, high_risk_alpha: np.ndarray) -> Image.Image:
    strong_arr = np.array(strong).astype(np.float32)
    safe_arr = np.array(safe.resize(strong.size)).astype(np.float32)
    if high_risk_alpha.shape[:2] != strong_arr.shape[:2]:
        high_risk_alpha = cv2.resize(high_risk_alpha, strong.size, interpolation=cv2.INTER_LINEAR)[..., None]
    out = strong_arr * (1.0 - high_risk_alpha) + safe_arr * high_risk_alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def label_panel(image: Image.Image, label: str, height: int = 384) -> Image.Image:
    width = int(image.width * height / image.height)
    canvas = image.resize((width, height)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    pad = 6
    draw.rectangle((0, 0, bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2), fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return canvas


def make_grid(panels: list[tuple[str, Image.Image]], out_path: Path) -> None:
    images = [(label, label_panel(image, label)) for label, image in panels]
    gutter = 8
    height = images[0][1].height
    grid = Image.new("RGB", (sum(image.width for _, image in images) + gutter * (len(images) - 1), height), (32, 32, 32))
    x = 0
    for _, image in images:
        grid.paste(image, (x, 0))
        x += image.width + gutter
    grid.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--style-sweep-run", default="city_forest_strength_20")
    parser.add_argument("--routing-run", default="debug_v0_adjusted")
    parser.add_argument("--strong-id", default="max_style_weak_structure")
    parser.add_argument("--safe-id", default="strong_style")
    parser.add_argument("--threshold", type=float, default=0.42)
    parser.add_argument("--softness", type=float, default=0.28)
    parser.add_argument("--run-name", default="city_forest_masked_twopass")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sweep_case = root / "runs" / "style_sweep" / args.style_sweep_run / args.case_id
    routing_case = root / "runs" / "routing_v0" / args.routing_run / args.case_id
    out_dir = root / "runs" / "masked_twopass_v0" / args.run_name / args.case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    content = load_rgb(sweep_case / args.safe_id / "content.png")
    style = load_rgb(sweep_case / args.safe_id / "style.png")
    safe = load_rgb(sweep_case / args.safe_id / "output.png")
    strong = load_rgb(sweep_case / args.strong_id / "output.png")
    risk = load_rgb(routing_case / "risk_map.png")

    alpha = risk_mask_from_heatmap(routing_case / "risk_map.png", args.threshold, args.softness)
    alpha_img = Image.fromarray(np.clip(alpha[..., 0] * 255, 0, 255).astype(np.uint8)).convert("RGB")
    result = composite(strong, safe, alpha)

    content.save(out_dir / "content.png")
    style.save(out_dir / "style.png")
    safe.save(out_dir / "safe.png")
    strong.save(out_dir / "strong.png")
    risk.save(out_dir / "risk_map.png")
    alpha_img.save(out_dir / "high_risk_alpha.png")
    result.save(out_dir / "masked_twopass.png")
    make_grid(
        [
            ("content", content),
            ("style", style),
            (args.safe_id, safe),
            ("masked_twopass", result),
            (args.strong_id, strong),
            ("risk_alpha", alpha_img),
        ],
        out_dir / "masked_twopass_grid.png",
    )
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(out_dir / "masked_twopass_grid.png")


if __name__ == "__main__":
    main()
