"""Capture the local runtime and repository state for the external benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def package_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", ""))
    except ImportError:
        return "not-installed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="external_benchmark/manifests/environment/sst_env.json",
    )
    parser.add_argument("--environment-id", default="local")
    args = parser.parse_args()

    environment = {
        "environment_id": args.environment_id,
        "environment_manager": "uv" if shutil.which("uv") else "unknown",
        "uv_version": command("uv", "--version"),
        "python": sys.version,
        "platform": platform.platform(),
        "repository_commit": command("git", "rev-parse", "HEAD"),
        "repository_status": command("git", "status", "--short"),
        "packages": {
            "torch": package_version("torch"),
            "diffusers": package_version("diffusers"),
            "transformers": package_version("transformers"),
            "PIL": package_version("PIL"),
            "cv2": package_version("cv2"),
        },
    }
    lockfile = ROOT / "uv.lock"
    environment["uv_lock_sha256"] = (
        hashlib.sha256(lockfile.read_bytes()).hexdigest() if lockfile.exists() else None
    )
    try:
        import torch

        environment["torch_cuda_version"] = torch.version.cuda
        environment["cuda_available"] = bool(torch.cuda.is_available())
        environment["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        environment["torch_cuda_version"] = None
        environment["cuda_available"] = False
        environment["cuda_device"] = None

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
