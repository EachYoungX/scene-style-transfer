import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = {
    "sd15": {
        "repo_id": "runwayml/stable-diffusion-v1-5",
        "note": "Use variant='fp16' when loading: from_pretrained(..., variant='fp16')",
        "allow_patterns": [
            "model_index.json",
            "feature_extractor/*",
            "safety_checker/config.json",
            "safety_checker/model.fp16.safetensors",
            "scheduler/*",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "tokenizer/*",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ],
    },
    "controlnet_canny": {
        "repo_id": "lllyasviel/control_v11p_sd15_canny",
        "note": "Use variant='fp16' when loading: from_pretrained(..., variant='fp16')",
        "allow_patterns": [
            "config.json",
            "*.fp16.safetensors",
        ],
    },
    "ip_adapter": {
        "repo_id": "h94/IP-Adapter",
        "note": "Only SD1.5 weights are downloaded.",
        "allow_patterns": [
            "models/image_encoder/config.json",
            "models/image_encoder/model.safetensors",
            "models/ip-adapter_sd15.safetensors",
        ],
    },
}


def download_model(name: str, config: dict, models_dir: Path) -> Path:
    target_dir = models_dir / name
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[SKIP] {name}: already exists at {target_dir}")
        return target_dir

    print(f"[DOWNLOAD] {name} from {config['repo_id']}")
    snapshot_download(
        repo_id=config["repo_id"],
        local_dir=str(target_dir),
        allow_patterns=config.get("allow_patterns"),
        ignore_patterns=["*.msgpack", "*.h5", "*.bin"],
    )
    print(f"[OK] {name} -> {target_dir}")
    return target_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--model", choices=list(MODELS.keys()))
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name, cfg in MODELS.items():
            print(f"  {name}: {cfg['repo_id']}  ({cfg.get('note', '')})")
        return

    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    targets = {args.model: MODELS[args.model]} if args.model else MODELS
    for name, config in targets.items():
        download_model(name, config, models_dir)

    print("\nAll downloads complete.")
    print("NOTE: When loading SD1.5 / ControlNet, pass variant='fp16' to from_pretrained().")


if __name__ == "__main__":
    main()
