"""Run the 23-pair seed42 generation-free feature validation for V2.4c."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from analyze_v2_4b_generation_free_features import (
        DEFAULT_CLIP_MODEL,
        DEFAULT_DINO_MODEL,
        load_representation_cache,
        representation_pair_features,
    )
    from build_v2_4_pair_preflight_analysis import (
        ROOT,
        aggregate_human,
        fit_square_crop,
        numeric,
        pair_features,
        profile_label,
        read_csv,
        spearman,
        write_csv,
    )
except ModuleNotFoundError:
    from scripts.analyze_v2_4b_generation_free_features import (
        DEFAULT_CLIP_MODEL,
        DEFAULT_DINO_MODEL,
        load_representation_cache,
        representation_pair_features,
    )
    from scripts.build_v2_4_pair_preflight_analysis import (
        ROOT,
        aggregate_human,
        fit_square_crop,
        numeric,
        pair_features,
        profile_label,
        read_csv,
        spearman,
        write_csv,
    )


V15_MANIFEST = ROOT / "configs/experiment/v1_5_cases.csv"
V23_MANIFEST = ROOT / "configs/experiment/v2_3_pair_response_profiles.csv"
V24B_MANIFEST = ROOT / "configs/experiment/v2_4b_targeted_profile_candidates.csv"
V22_LABELS = ROOT / "runs/ip_adapter_plus_injection/v2_2a_safe_strength_frontier/audits/targeted_multiseed/human_sensitivity_annotations.csv"
V23_LABELS = ROOT / "runs/ip_adapter_plus_injection/v2_3_pair_response_profiles/audits/human_sensitivity_annotations.csv"
V24B_LABELS = ROOT / "runs/ip_adapter_plus_injection/v2_4b_targeted_profile_candidates/audits/human_sensitivity_annotations.csv"

V15_CASES = {"v1_5_demuth_church", "v1_5_kulhanek_snow_winter", "v1_5_demuth_wave"}
V15_TO_V22 = {
    "v1_5_demuth_church": "v1_5_demuth_church",
    "v1_5_kulhanek_snow_winter": "v1_5_kulhanek_snow_winter",
    "v1_5_demuth_wave": "v1_5_demuth_wave",
}

TARGETS = (
    "style_valid",
    "baseline_takeover_02",
    "style_gain_if_valid",
    "incremental_takeover_max",
    "incremental_nonzero_count",
    "late_escalation",
)


def content_family(path: str) -> str:
    name = Path(path).stem
    for token, family in (
        ("church", "church"),
        ("lake", "lake"),
        ("forest", "forest"),
        ("wave", "wave"),
        ("coast", "coast"),
        ("street", "city"),
        ("snow", "snow"),
    ):
        if token in name:
            return family
    return name


def reference_family(path: str) -> str:
    return Path(path).parent.name


def common_pair_manifest() -> list[dict[str, str]]:
    v15 = {row["case_id"]: row for row in read_csv(V15_MANIFEST)}
    v23 = read_csv(V23_MANIFEST)
    candidates = read_csv(V24B_MANIFEST)
    rows: list[dict[str, str]] = []
    for case_id in V15_CASES:
        source = v15[case_id]
        rows.append(
            {
                "case": case_id,
                "content_path": source["content"],
                "reference_path": source["style"],
                "content_family": content_family(source["content"]),
                "reference_family": reference_family(source["style"]),
                "label_source": "v2.2a seed42",
            }
        )
    for source in v23:
        rows.append(
            {
                "case": source["canonical_case_id"],
                "content_path": source["content_path"],
                "reference_path": source["style_path"],
                "content_family": content_family(source["content_path"]),
                "reference_family": reference_family(source["style_path"]),
                "label_source": "v2.3 seed42",
            }
        )
    for source in candidates:
        rows.append(
            {
                "case": source["case_id"],
                "content_path": source["content_path"],
                "reference_path": source["style_path"],
                "content_family": source["content_family"],
                "reference_family": source["reference_family"],
                "label_source": "v2.4b seed42",
            }
        )
    if len(rows) != 23 or len({row["case"] for row in rows}) != 23:
        raise ValueError(f"Expected 23 unique pairs, got {len(rows)}")
    return rows


def seed42_rows(path: Path) -> list[dict[str, str]]:
    return [row for row in read_csv(path) if row.get("seed", "42") == "42"]


def normalize_v22(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized = []
    for row in rows:
        lam = numeric(row.get("lambda"))
        takeover = row.get("human_takeover_score_0_3", "")
        if lam is None:
            continue
        normalized.append(
            {
                "case": row["case"],
                "seed": row["seed"],
                "lambda": row["lambda"],
                "human_style_score_0_4": row.get("human_style_score_0_4", ""),
                "baseline_takeover_0_3": takeover if lam == 0.2 else "NA",
                "incremental_takeover_0_3": "NA" if lam == 0.2 else takeover,
                "style_valid": "true",
            }
        )
    return normalized


def load_common_labels() -> dict[str, list[dict[str, str]]]:
    labels: dict[str, list[dict[str, str]]] = {}
    for row in normalize_v22(seed42_rows(V22_LABELS)):
        labels.setdefault(V15_TO_V22.get(row["case"], row["case"]), []).append(row)
    for row in seed42_rows(V23_LABELS) + seed42_rows(V24B_LABELS):
        labels.setdefault(row["case"], []).append(row)
    return labels


def pair_profile(rows: list[dict[str, str]]) -> dict[str, object]:
    profile = aggregate_human(rows)
    valid = int(any(row.get("style_valid", "").lower() == "true" for row in rows))
    profile["style_valid"] = valid
    profile["style_gain_if_valid"] = profile["style_gain_median"] if valid else ""
    profile["baseline_takeover_02"] = profile["baseline_takeover_median"]
    profile["incremental_takeover_max"] = profile["incremental_takeover_max_median"]
    profile["incremental_nonzero_count"] = profile["incremental_nonzero_interval_count"]
    profile["late_escalation"] = profile["late_escalation_frequency"]
    return profile


def target_value(row: dict[str, object], target: str) -> float | None:
    if target == "style_valid":
        return float(row["style_valid"])
    return numeric(row.get(target))


def cliffs_delta(low: list[float], high: list[float]) -> float | str:
    if not low or not high:
        return ""
    comparisons = [np.sign(high_value - low_value) for high_value in high for low_value in low]
    return float(np.mean(comparisons))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-model", default=str(DEFAULT_CLIP_MODEL))
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", default="analysis")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir

    pair_rows = common_pair_manifest()
    labels = load_common_labels()
    profiles = {case: pair_profile(values) for case, values in labels.items()}
    if set(profiles) != {row["case"] for row in pair_rows}:
        missing = sorted({row["case"] for row in pair_rows} - set(profiles))
        raise ValueError(f"Missing common-seed labels for: {missing}")

    merged = []
    base_features_by_case = {}
    paths: set[Path] = set()
    for pair in pair_rows:
        content_path = ROOT / pair["content_path"]
        reference_path = ROOT / pair["reference_path"]
        base_features_by_case[pair["case"]] = pair_features(content_path, reference_path)
        paths.update((content_path, reference_path))
    cache, model_manifest = load_representation_cache(
        sorted(paths),
        Path(args.clip_model) if args.clip_model else None,
        args.dino_model or None,
        args.device,
        args.batch_size,
    )

    for pair in pair_rows:
        case = pair["case"]
        content_path = ROOT / pair["content_path"]
        reference_path = ROOT / pair["reference_path"]
        merged.append(
            {
                **pair,
                **profiles[case],
                "profile_label": profile_label(profiles[case]),
                **base_features_by_case[case],
                **representation_pair_features(content_path, reference_path, cache),
                "feature_source": "512px fit_square_crop; Canny/LSD; RGB patch; local CLIP vision; DINOv2-small",
                "seed_scope": "seed42_common_screening",
                "generation": False,
            }
        )

    non_feature = {
        "case",
        "content_path",
        "reference_path",
        "content_family",
        "reference_family",
        "label_source",
        "seed_scope",
        "feature_source",
        "generation",
        "profile_label",
        "label_status",
    }
    feature_fields = [
        field
        for field in merged[0]
        if field not in non_feature
        and field not in TARGETS
        and field not in {"seed_count", "style_valid_rate"}
        and not field.endswith("_median")
        and not field.endswith("_max")
        and not field.endswith("_frequency")
        and not field.endswith("_count")
        and not field.endswith("_02")
        and not field.endswith("_if_valid")
        and not field.endswith("_status")
        and not field.endswith("_takeover")
        and not field.endswith("_gain")
    ]
    feature_fields = [field for field in feature_fields if numeric(merged[0].get(field)) is not None]

    write_csv(output_dir / "v2_4c_common_seed_profiles.csv", merged)
    correlations = []
    for target in TARGETS:
        target_rows = merged if target != "style_gain_if_valid" else [row for row in merged if row["style_valid"] == 1]
        for feature in feature_fields:
            pairs = [(target_value(row, target), numeric(row.get(feature))) for row in target_rows]
            pairs = [(target_value_, feature_value) for target_value_, feature_value in pairs if target_value_ is not None and feature_value is not None]
            correlations.append(
                {
                    "target": target,
                    "subset": "all_23_seed42" if target != "style_gain_if_valid" else "style_valid_true_only",
                    "feature": feature,
                    "n": len(pairs),
                    "spearman_rho": spearman([pair[1] for pair in pairs], [pair[0] for pair in pairs]),
                }
            )
    write_csv(output_dir / "v2_4c_common_seed_correlations.csv", correlations)

    effects = []
    for target, positive in (("style_valid", 1), ("baseline_takeover_02", 2), ("late_escalation", 1)):
        low_rows = [row for row in merged if (target_value(row, target) or 0) == (0 if target != "baseline_takeover_02" else 0)]
        high_rows = [row for row in merged if (target_value(row, target) or 0) >= positive]
        for feature in feature_fields:
            low = [numeric(row.get(feature)) for row in low_rows]
            high = [numeric(row.get(feature)) for row in high_rows]
            low = [value for value in low if value is not None]
            high = [value for value in high if value is not None]
            if low and high:
                effects.append(
                    {
                        "target": target,
                        "feature": feature,
                        "n_low": len(low),
                        "n_high": len(high),
                        "median_low": float(np.median(low)),
                        "median_high": float(np.median(high)),
                        "high_minus_low": float(np.median(high) - np.median(low)),
                        "cliffs_delta_high_vs_low": cliffs_delta(low, high),
                    }
                )
    write_csv(output_dir / "v2_4c_common_seed_effects.csv", effects)
    (output_dir / "v2_4c_common_seed_manifest.json").write_text(
        json.dumps(
            {
                "pair_count": len(merged),
                "seed_scope": "seed42_common_screening",
                "targets": list(TARGETS),
                "feature_count": len(feature_fields),
                "feature_fields": feature_fields,
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
