"""Inspect whether the current IP-Adapter setup exposes local reference tokens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from run_baseline import fit_square  # noqa: E402

from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", default="data/raw/hokusai/hokusai_great_wave_1831.jpg")
    parser.add_argument("--model-dir", default="models/sd15")
    parser.add_argument("--controlnet-dir", default="models/controlnet_canny")
    parser.add_argument("--ip-adapter-dir", default="models/ip_adapter")
    parser.add_argument("--weight-name", default="ip-adapter_sd15.safetensors")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    controlnet = ControlNetModel.from_pretrained(
        root / args.controlnet_dir,
        torch_dtype=torch.float16,
        local_files_only=True,
        variant="fp16",
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        root / args.model_dir,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
        variant="fp16",
    )
    pipe.load_ip_adapter(
        str(root / args.ip_adapter_dir),
        subfolder="models",
        weight_name=args.weight_name,
        local_files_only=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    style = fit_square(root / args.style, 512)
    embeds = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=style,
        ip_adapter_image_embeds=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

    print("Prepared IP-Adapter image embeds:")
    for i, embed in enumerate(embeds):
        print(f"  embed[{i}]: shape={tuple(embed.shape)} dtype={embed.dtype}")

    print("\nUNet encoder_hid_proj:")
    print(f"  class={pipe.unet.encoder_hid_proj.__class__.__name__}")
    if hasattr(pipe.unet.encoder_hid_proj, "image_projection_layers"):
        for i, layer in enumerate(pipe.unet.encoder_hid_proj.image_projection_layers):
            print(f"  projection[{i}]: class={layer.__class__.__name__}")
            for attr in ["num_image_text_embeds", "num_queries", "clip_embeddings_dim", "cross_attention_dim"]:
                if hasattr(layer, attr):
                    print(f"    {attr}={getattr(layer, attr)}")

    has_patch_tokens = bool(embeds) and embeds[0].ndim >= 4 and embeds[0].shape[-2] > 1
    has_sequence_tokens = bool(embeds) and embeds[0].ndim == 3 and embeds[0].shape[1] > 1
    if not has_patch_tokens and not has_sequence_tokens:
        print("\nConclusion: current prepared reference embedding is pooled/global, not a spatial patch-token bank.")
        print("Token-level shuffling or local reference-structure suppression is not supported by this base setup.")
    else:
        print("\nConclusion: multiple reference tokens are available; V1-A token manipulation is feasible.")


if __name__ == "__main__":
    main()
