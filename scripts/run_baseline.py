"""Run baseline diffusion style-transfer pipelines.

Examples:
  python scripts/run_baseline.py --method img2img --content data/raw/_photo_ref/photo_landscape_mountains_with_lake.jpg
  python scripts/run_baseline.py --method controlnet_canny --content data/raw/_photo_ref/photo_dim_city_building.jpg
  python scripts/run_baseline.py --method ip_adapter_canny --content data/raw/_photo_ref/photo_dim_city_building.jpg --style data/raw/monet/monet_rouen_cathedral_1894.jpg
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image

from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
)


Method = Literal["img2img", "controlnet_canny", "ip_adapter_canny"]


@dataclass(frozen=True)
class RunConfig:
    method: Method
    content: str
    style: str | None
    prompt: str
    negative_prompt: str
    seed: int
    num_inference_steps: int
    guidance_scale: float
    strength: float
    controlnet_scale: float
    ip_adapter_scale: float
    size: int
    model_dir: str
    controlnet_dir: str
    ip_adapter_dir: str


METHODS: tuple[Method, ...] = ("img2img", "controlnet_canny", "ip_adapter_canny")


def fit_square(image_path: Path, size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def make_canny(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    edges = cv2.Canny(np.array(image), low_threshold, high_threshold)
    return Image.fromarray(np.stack([edges, edges, edges], axis=-1))


def make_output_dir(root: Path, method: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = root / "runs" / "baselines" / method / stamp
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def load_img2img_pipeline(model_dir: Path) -> StableDiffusionImg2ImgPipeline:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
        variant="fp16",
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe.to("cuda")


def load_controlnet_pipeline(
    model_dir: Path,
    controlnet_dir: Path,
    ip_adapter_dir: Path | None = None,
    ip_adapter_scale: float = 0.5,
) -> StableDiffusionControlNetImg2ImgPipeline:
    controlnet = ControlNetModel.from_pretrained(
        controlnet_dir,
        torch_dtype=torch.float16,
        local_files_only=True,
        variant="fp16",
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model_dir,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
        variant="fp16",
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if ip_adapter_dir is not None:
        pipe.load_ip_adapter(
            str(ip_adapter_dir),
            subfolder="models",
            weight_name="ip-adapter_sd15.safetensors",
            local_files_only=True,
        )
        pipe.set_ip_adapter_scale(ip_adapter_scale)
        # Do not enable attention slicing with IP-Adapter: current processors pass tuple encoder states.

    return pipe.to("cuda")


def run(config: RunConfig, project_root: Path, output_dir: Path | None = None) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these baseline runs.")

    content_path = project_root / config.content
    style_path = project_root / config.style if config.style else None
    if not content_path.exists():
        raise FileNotFoundError(content_path)
    if config.method == "ip_adapter_canny" and style_path is None:
        raise ValueError("--style is required for method ip_adapter_canny")
    if style_path is not None and not style_path.exists():
        raise FileNotFoundError(style_path)

    out_dir = output_dir if output_dir is not None else make_output_dir(project_root, config.method)
    out_dir.mkdir(parents=True, exist_ok=False)
    content = fit_square(content_path, config.size)
    content.save(out_dir / "content.png")

    control_image = None
    if config.method in {"controlnet_canny", "ip_adapter_canny"}:
        control_image = make_canny(content)
        control_image.save(out_dir / "canny.png")

    style = None
    if style_path is not None:
        style = fit_square(style_path, config.size)
        style.save(out_dir / "style.png")

    model_dir = project_root / config.model_dir
    controlnet_dir = project_root / config.controlnet_dir
    ip_adapter_dir = project_root / config.ip_adapter_dir

    if config.method == "img2img":
        pipe = load_img2img_pipeline(model_dir)
    elif config.method == "controlnet_canny":
        pipe = load_controlnet_pipeline(model_dir, controlnet_dir)
    elif config.method == "ip_adapter_canny":
        pipe = load_controlnet_pipeline(model_dir, controlnet_dir, ip_adapter_dir, config.ip_adapter_scale)
    else:
        raise ValueError(f"Unknown method: {config.method}")

    generator = torch.Generator(device="cuda").manual_seed(config.seed)
    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    kwargs = {
        "prompt": config.prompt,
        "negative_prompt": config.negative_prompt,
        "image": content,
        "strength": config.strength,
        "guidance_scale": config.guidance_scale,
        "num_inference_steps": config.num_inference_steps,
        "generator": generator,
    }
    if control_image is not None:
        kwargs["control_image"] = control_image
        kwargs["controlnet_conditioning_scale"] = config.controlnet_scale
    if style is not None:
        kwargs["ip_adapter_image"] = style

    result = pipe(**kwargs).images[0]
    elapsed_sec = time.time() - start
    peak_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3

    result.save(out_dir / "output.png")
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    metrics = {
        "elapsed_sec": round(elapsed_sec, 4),
        "peak_allocated_gb": round(peak_allocated_gb, 4),
        "cuda_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    (out_dir / "runtime.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return out_dir


def parse_args() -> tuple[RunConfig, str | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--content", required=True)
    parser.add_argument("--style")
    parser.add_argument("--prompt", default="a coherent scene with painterly atmosphere")
    parser.add_argument("--negative-prompt", default="low quality, blurry, distorted, text, watermark, copied objects")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=6.5)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--controlnet-scale", type=float, default=0.8)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.45)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--model-dir", default="models/sd15")
    parser.add_argument("--controlnet-dir", default="models/controlnet_canny")
    parser.add_argument("--ip-adapter-dir", default="models/ip_adapter")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = RunConfig(
        method=args.method,
        content=args.content,
        style=args.style,
        prompt=args.prompt,
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
    return config, args.output_dir


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config, output_dir_arg = parse_args()
    output_dir = project_root / output_dir_arg if output_dir_arg else None
    out_dir = run(config, project_root, output_dir)
    print(f"Saved run to {out_dir}")


if __name__ == "__main__":
    main()
