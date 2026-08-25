"""Run V2.4b representation features and pair-level hypothesis checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from build_v2_4_pair_preflight_analysis import ROOT, fit_square_crop, numeric, read_csv, write_csv
except ModuleNotFoundError:
    from scripts.build_v2_4_pair_preflight_analysis import ROOT, fit_square_crop, numeric, read_csv, write_csv


PROFILE_PATH = ROOT / "analysis/v2_4_pair_profiles.csv"
FEATURE_PATH = ROOT / "analysis/v2_4_pair_features.csv"
DEFAULT_CLIP_MODEL = ROOT / "models/ip_adapter_plus/models/image_encoder"
DEFAULT_DINO_MODEL = "facebook/dinov2-small"
DEMUTH_CASES = {
    "v1_5_demuth_church",
    "clean_demuth_G1_water_lake",
    "clean_demuth_G1_forest",
    "clean_demuth_G4_city_mismatch",
}
TARGET_FIELDS = (
    "baseline_takeover_median",
    "style_gain_median",
    "incremental_takeover_max_median",
    "incremental_nonzero_interval_count",
    "late_escalation_frequency",
)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def nearest_patch_features(content: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    similarity = normalize_rows(content) @ normalize_rows(reference).T
    forward = similarity.max(axis=1)
    forward_argmax = similarity.argmax(axis=1)
    backward_argmax = similarity.argmax(axis=0)
    mutual = float(np.mean([backward_argmax[target] == index for index, target in enumerate(forward_argmax)]))
    return {
        f"{prefix}_patch_nearest_mean": float(forward.mean()),
        f"{prefix}_patch_nearest_p10": float(np.percentile(forward, 10)),
        f"{prefix}_patch_mutual_nearest_fraction": mutual,
    }


def load_representation_cache(
    image_paths: list[Path],
    clip_model: Path | None,
    dino_model: str | None,
    device_name: str,
    batch_size: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    import torch
    from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPVisionModelWithProjection

    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    cache: dict[str, dict[str, np.ndarray]] = {str(path): {} for path in image_paths}
    manifest: dict[str, object] = {"device": str(device)}

    def run_model(kind: str, model, processor) -> None:
        model.to(device).eval()
        for start in range(0, len(image_paths), batch_size):
            paths = image_paths[start : start + batch_size]
            images = [Image.fromarray(fit_square_crop(path)) for path in paths]
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.inference_mode():
                outputs = model(pixel_values=pixel_values)
            tokens = outputs.last_hidden_state.detach().float().cpu().numpy()
            if kind == "clip":
                global_values = outputs.image_embeds.detach().float().cpu().numpy()
            else:
                global_values = tokens[:, 0, :]
            for index, path in enumerate(paths):
                cache[str(path)][f"{kind}_global"] = normalize_rows(global_values[[index]])[0]
                cache[str(path)][f"{kind}_patches"] = normalize_rows(tokens[index, 1:, :])

    if clip_model is not None:
        if not clip_model.exists():
            raise FileNotFoundError(f"CLIP checkpoint not found: {clip_model}")
        clip_processor = CLIPImageProcessor(
            size={"shortest_edge": 224},
            crop_size={"height": 224, "width": 224},
            rescale_factor=1 / 255,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
        )
        clip = CLIPVisionModelWithProjection.from_pretrained(clip_model, local_files_only=True)
        run_model("clip", clip, clip_processor)
        manifest["clip_model"] = str(clip_model)
    else:
        manifest["clip_model"] = None

    if dino_model:
        dino_processor = AutoImageProcessor.from_pretrained(dino_model)
        dino = AutoModel.from_pretrained(dino_model)
        run_model("dino", dino, dino_processor)
        manifest["dino_model"] = dino_model
    else:
        manifest["dino_model"] = None
    return cache, manifest


def representation_pair_features(
    content_path: Path,
    reference_path: Path,
    cache: dict[str, dict[str, np.ndarray]],
) -> dict[str, float | str]:
    content = cache.get(str(content_path), {})
    reference = cache.get(str(reference_path), {})
    output: dict[str, float | str] = {}
    for kind in ("clip", "dino"):
        global_key = f"{kind}_global_cosine"
        patch_values = (content.get(f"{kind}_patches"), reference.get(f"{kind}_patches"))
        content_global = content.get(f"{kind}_global")
        reference_global = reference.get(f"{kind}_global")
        output[global_key] = float(content_global @ reference_global) if content_global is not None and reference_global is not None else ""
        if patch_values[0] is not None and patch_values[1] is not None:
            output.update(nearest_patch_features(patch_values[0], patch_values[1], kind))
        else:
            output.update(
                {
                    f"{kind}_patch_nearest_mean": "",
                    f"{kind}_patch_nearest_p10": "",
                    f"{kind}_patch_mutual_nearest_fraction": "",
                }
            )
    return output


def average_ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = np.asarray(values)[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float | str:
    if len(x) < 3:
        return ""
    x_rank = average_ranks(x)
    y_rank = average_ranks(y)
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return ""
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def cliffs_delta(low: list[float], high: list[float]) -> float | str:
    if not low or not high:
        return ""
    comparisons = [np.sign(high_value - low_value) for high_value in high for low_value in low]
    return float(np.mean(comparisons))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-model", default=str(DEFAULT_CLIP_MODEL))
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", default="analysis")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir

    profiles = read_csv(PROFILE_PATH)
    base_features = read_csv(FEATURE_PATH)
    feature_by_case = {row["case"]: row for row in base_features}
    paths = sorted({Path(row["content_path"]) for row in base_features} | {Path(row["reference_path"]) for row in base_features})
    paths = [ROOT / path if not path.is_absolute() else path for path in paths]
    cache, model_manifest = load_representation_cache(
        paths,
        Path(args.clip_model) if args.clip_model else None,
        args.dino_model or None,
        args.device,
        args.batch_size,
    )

    merged = []
    representation_fields: set[str] = set()
    for profile in profiles:
        base = feature_by_case[profile["case"]]
        content_path = ROOT / base["content_path"]
        reference_path = ROOT / base["reference_path"]
        representation = representation_pair_features(content_path, reference_path, cache)
        representation_fields.update(representation)
        merged.append(
            {
                **profile,
                **base,
                "feature_source": "512px fit_square_crop; Canny/LSD; RGB patch baseline; local CLIP vision; DINOv2-small",
                **representation,
            }
        )
    base_feature_fields = [
        field
        for field in base_features[0]
        if field not in {"case", "content_path", "reference_path", "feature_source"}
        and not field.endswith("_histogram")
    ]
    feature_fields = [
        *base_feature_fields,
        *sorted(representation_fields),
    ]
    write_csv(output_dir / "v2_4_pair_feature_analysis.csv", merged)

    correlations = []
    for target in TARGET_FIELDS:
        for feature in feature_fields:
            pairs = [(numeric(row.get(target)), numeric(row.get(feature))) for row in merged]
            pairs = [(target_value, feature_value) for target_value, feature_value in pairs if target_value is not None and feature_value is not None]
            correlations.append(
                {
                    "target": target,
                    "feature": feature,
                    "n": len(pairs),
                    "spearman_rho": spearman([pair[1] for pair in pairs], [pair[0] for pair in pairs]),
                }
            )
    write_csv(output_dir / "v2_4_pair_feature_correlations.csv", correlations)

    low = [row for row in merged if numeric(row.get("baseline_takeover_median")) == 0]
    high = [row for row in merged if (numeric(row.get("baseline_takeover_median")) or -1) >= 2]
    effects = []
    for feature in feature_fields:
        low_values = [numeric(row.get(feature)) for row in low]
        high_values = [numeric(row.get(feature)) for row in high]
        low_values = [value for value in low_values if value is not None]
        high_values = [value for value in high_values if value is not None]
        if not low_values or not high_values:
            continue
        effects.append(
            {
                "target": "baseline_takeover_0_vs_ge2",
                "feature": feature,
                "n_low": len(low_values),
                "n_high": len(high_values),
                "median_low": float(np.median(low_values)),
                "median_high": float(np.median(high_values)),
                "high_minus_low": float(np.median(high_values) - np.median(low_values)),
                "cliffs_delta_high_vs_low": cliffs_delta(low_values, high_values),
            }
        )
    write_csv(output_dir / "v2_4_feature_effects.csv", effects)

    demuth_rows = [{key: row[key] for key in ("case", "profile_label", *TARGET_FIELDS, *feature_fields) if key in row} for row in merged if row["case"] in DEMUTH_CASES]
    write_csv(output_dir / "v2_4_demuth_controlled_subset.csv", demuth_rows)

    (output_dir / "v2_4b_feature_manifest.json").write_text(
        json.dumps(
            {
                "pair_count": len(merged),
                "feature_count": len(feature_fields),
                "representation_features": sorted(representation_fields),
                "target_fields": list(TARGET_FIELDS),
                "model": model_manifest,
                "generation": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
