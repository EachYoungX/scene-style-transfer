"""Create T0-T4 post-hoc blending diagnostics for a case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from run_masked_twopass_v0 import (
    composite,
    hard_dilated_risk_mask,
    label_panel,
    load_rgb,
    make_grid,
    oracle_corridor_mask,
    risk_mask_from_heatmap,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--style-sweep-run", default="city_forest_strength_20")
    parser.add_argument("--routing-run", default="debug_v0_adjusted")
    parser.add_argument("--safe-id", default="strong_style")
    parser.add_argument("--strong-id", default="max_style_weak_structure")
    parser.add_argument("--run-name", default="city_forest_twopass_diagnostic")
    parser.add_argument("--threshold", type=float, default=0.32)
    parser.add_argument("--dilation", type=int, default=35)
    parser.add_argument("--close", type=int, default=17)
    parser.add_argument("--blur", type=float, default=6.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sweep_case = root / "runs" / "style_sweep" / args.style_sweep_run / args.case_id
    routing_case = root / "runs" / "routing_v0" / args.routing_run / args.case_id
    out_dir = root / "runs" / "twopass_diagnostic" / args.run_name / args.case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    content = load_rgb(sweep_case / args.safe_id / "content.png")
    style = load_rgb(sweep_case / args.safe_id / "style.png")
    safe = load_rgb(sweep_case / args.safe_id / "output.png")
    strong = load_rgb(sweep_case / args.strong_id / "output.png")
    risk_path = routing_case / "risk_map.png"

    soft_alpha = risk_mask_from_heatmap(risk_path, threshold=0.42, softness=0.28)
    hard_alpha = hard_dilated_risk_mask(risk_path, args.threshold, args.dilation, args.close, args.blur)
    oracle_alpha = oracle_corridor_mask(strong.size, args.case_id, args.blur)
    half_alpha = np.full_like(soft_alpha, 0.5, dtype=np.float32)

    outputs = {
        "T0_strong_only": strong,
        "T1_soft_risk": composite(strong, safe, soft_alpha),
        "T2_dilated_risk": composite(strong, safe, hard_alpha),
        "T3_oracle_mask": composite(strong, safe, oracle_alpha),
        "T4_half_blend": composite(strong, safe, half_alpha),
    }
    alphas = {
        "T1_soft_alpha": soft_alpha,
        "T2_dilated_alpha": hard_alpha,
        "T3_oracle_alpha": oracle_alpha,
    }

    content.save(out_dir / "content.png")
    style.save(out_dir / "style.png")
    safe.save(out_dir / "safe.png")
    strong.save(out_dir / "strong.png")
    for name, image in outputs.items():
        image.save(out_dir / f"{name}.png")
    for name, alpha in alphas.items():
        Image.fromarray(np.clip(alpha[..., 0] * 255, 0, 255).astype(np.uint8)).convert("RGB").save(out_dir / f"{name}.png")

    make_grid(
        [
            ("content", content),
            ("style", style),
            ("safe", safe),
            ("T0 strong", outputs["T0_strong_only"]),
            ("T1 soft", outputs["T1_soft_risk"]),
            ("T2 dilated", outputs["T2_dilated_risk"]),
            ("T3 oracle", outputs["T3_oracle_mask"]),
            ("T4 half", outputs["T4_half_blend"]),
        ],
        out_dir / "diagnostic_grid.png",
    )
    make_grid(
        [
            ("T1 alpha", Image.open(out_dir / "T1_soft_alpha.png")),
            ("T2 alpha", Image.open(out_dir / "T2_dilated_alpha.png")),
            ("T3 alpha", Image.open(out_dir / "T3_oracle_alpha.png")),
        ],
        out_dir / "alpha_grid.png",
    )
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(out_dir / "diagnostic_grid.png")
    print(out_dir / "alpha_grid.png")


if __name__ == "__main__":
    main()
