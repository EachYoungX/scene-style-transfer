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
