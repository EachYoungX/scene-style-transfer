# External Same-Protocol Benchmark

This directory contains the frozen benchmark setup for the post-V2.4 comparison.

Current state:

- 24 held-out pairs are frozen in `configs/external_eval_pairs_v1.csv`.
- The protocol is frozen in `configs/protocol_freeze.yaml`.
- The three-track design is frozen in `configs/tracks.yaml`:
  - Track A: native-quality benchmark under each method's official pipeline.
  - Track B: controlled SD1.5 comparison for the injection-control question.
  - Track C: RTX 4060 Laptop 8GB feasibility and operational cost.
- The method roster is frozen in `configs/methods.yaml`.
- Primary human evaluation uses absolute style/content/risk scores; takeover rates are not final benchmark metrics.
- No final evaluation outputs have been generated.
- Python environments are managed with `uv`; the project environment uses the repository `uv.lock`, while method families use isolated environments under `.venvs/` and the shared uv cache.
- Core comparison: AdaIN, Vanilla IP-Adapter + same Canny, InstantStyle, MaskST (Less is More), StyleShot, CoCoDiff, and the frozen A2 method.
- StyleGallery is optional pending an RTX 4060 8GB smoke test.
- Puff-Net and StyleID are supplementary/deferred; StyleKeeper is deferred because its public repository does not expose a complete official inference path.
- EFDM was added as the deterministic traditional baseline. The official decoder supplied locally was validated with the official PyTorch test path, and all 24 held-out pairs generated valid 512x512 outputs under Track A. The compatible VGG file comes from the official AdaIN release because the EFDM README's VGG Drive link is currently unavailable.
- Z-STAR was checked with the official repository and the existing local SD1.5 checkpoint. A clean-boot 512x512 single-pair, 2-step run completed content and style inversion but failed during synthesis with CUDA out-of-memory; its process reached about 14.1 GiB on an 8 GiB GPU. It is excluded from the active tracks, and no further retry is planned.
- StyleID received the final clean-boot bounded attempt. The SD1.5 Diffusers path loaded, then failed at VAE encoding with CUDA out-of-memory (requested allocation about 16.88 GiB). It is excluded from the active tracks, and no further retry is planned.
- CAST (ACCV 2024, VQ autoencoder) has a verifiable paper and author publication entry, but no official implementation or checkpoint download was exposed by those sources. It remains a literature candidate, not a reproducible comparator.
- AdaIN is blocked by its Torch7 runtime; CoCoDiff is blocked pending its separate legacy environment and checkpoints. Other diffusion methods still require official checkpoints and smoke validation.
- StyleShot's public repository currently identifies the implementation with its arXiv record; the claimed TPAMI venue must be verified before manuscript finalization.

The 23-pair V2.4 development set is not part of the final held-out manifest.

Track A and Track B share the held-out pairs, single-reference setup, 512x512 output, and frozen global calibration rule. Track C records OOM/unavailable-runtime outcomes instead of modifying an official method to force an 8GB result.

Environment workflow:

```bash
uv run pytest -q
uv venv external_benchmark/.venvs/maskst_styleshot_uv --python 3.10
uv pip install --python external_benchmark/.venvs/maskst_styleshot_uv/bin/python -r external_benchmark/vendor/maskst/requirements.txt
```

Use the family-specific entries in `configs/environments.yaml`. Do not install external pinned dependencies into the project `.venv`; record the environment path, resolved packages, CUDA/Torch versions, and the exact official commit in the environment manifest before smoke generation.
