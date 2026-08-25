"""构建 V2.4 的 pair profile 与生成前兼容性特征。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SIZE = 512
V23_CASES = ROOT / "configs/experiment/v2_3_pair_response_profiles.csv"
V15_CASES = ROOT / "configs/experiment/v1_5_cases.csv"
V23_HUMAN = ROOT / "runs/ip_adapter_plus_injection/v2_3_pair_response_profiles/audits/representative_multiseed/human_sensitivity_annotations.csv"
V23_HUMAN_SEED42 = ROOT / "runs/ip_adapter_plus_injection/v2_3_pair_response_profiles/audits/human_sensitivity_annotations.csv"
V22_CANONICAL_HUMAN = ROOT / "runs/ip_adapter_plus_injection/v2_2a_safe_strength_frontier/audits/targeted_multiseed/human_sensitivity_annotations.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open(newline="", encoding=encoding) as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法读取 CSV：{path}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"没有可写入的数据：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fit_square_crop(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    side = min(image.width, image.height)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def percentile(values: np.ndarray, value: float) -> float:
    return float(np.percentile(values, value)) if values.size else 0.0


def edge_features(image: np.ndarray) -> dict[str, object]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    line_density = 0.0
    lengths = np.empty(0, dtype=np.float32)
    orientation_hist = np.zeros(18, dtype=np.float64)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = detector.detect(gray)[0]
    if lines is not None:
        segments = np.asarray(lines, dtype=np.float32).reshape(-1, 4)
        dx = segments[:, 2] - segments[:, 0]
        dy = segments[:, 3] - segments[:, 1]
        lengths = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        angles = np.mod(np.arctan2(dy, dx), np.pi)
        bins = np.floor(angles / np.pi * 18).astype(int).clip(0, 17)
        for index, length in zip(bins, lengths):
            orientation_hist[index] += float(length)
        line_density = float(lengths.sum() / (SIZE * SIZE))
    probabilities = orientation_hist / max(orientation_hist.sum(), 1e-12)
    entropy = float(-(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])).sum() / np.log(18))
    return {
        "canny_density": float(edges.mean() / 255.0),
        "lsd_line_density": line_density,
        "lsd_length_mean": float(lengths.mean()) if lengths.size else 0.0,
        "lsd_length_p90": percentile(lengths, 90),
        "lsd_orientation_entropy": entropy,
        "lsd_orientation_max_share": float(probabilities.max()),
        "_orientation_distribution": probabilities,
    }


def appearance_features(image: np.ndarray) -> dict[str, float]:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]).astype(np.float32)
    hist /= max(float(hist.sum()), 1e-12)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(gray - gray.mean())))
    yy, xx = np.ogrid[:SIZE, :SIZE]
    radius = np.sqrt((xx - SIZE / 2) ** 2 + (yy - SIZE / 2) ** 2)
    high_frequency = float(spectrum[radius > SIZE * 0.25].sum() / max(spectrum.sum(), 1e-12))
    return {
        "lab_mean_l": float(lab[:, :, 0].mean()),
        "lab_mean_a": float(lab[:, :, 1].mean()),
        "lab_mean_b": float(lab[:, :, 2].mean()),
        "lab_std_l": float(lab[:, :, 0].std()),
        "lab_std_a": float(lab[:, :, 1].std()),
        "lab_std_b": float(lab[:, :, 2].std()),
        "hsv_histogram": hist,
        "contrast_std": float(gray.std()),
        "laplacian_variance": float(laplacian.var()),
        "high_frequency_ratio": high_frequency,
    }


def patch_descriptors(image: np.ndarray, grid: int = 16) -> np.ndarray:
    cell = SIZE // grid
    descriptors = []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    for row in range(grid):
        for col in range(grid):
            patch = image[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell].astype(np.float32) / 255.0
            patch_gray = gray[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell]
            descriptors.append(np.concatenate([patch.mean((0, 1)), patch.std((0, 1)), [patch_gray.mean(), patch_gray.std()]]))
    return np.asarray(descriptors, dtype=np.float32)


def coarse_patch_features(content: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    content_desc = patch_descriptors(content)
    reference_desc = patch_descriptors(reference)
    content_desc /= np.maximum(np.linalg.norm(content_desc, axis=1, keepdims=True), 1e-8)
    reference_desc /= np.maximum(np.linalg.norm(reference_desc, axis=1, keepdims=True), 1e-8)
    similarity = content_desc @ reference_desc.T
    forward = similarity.max(axis=1)
    forward_argmax = similarity.argmax(axis=1)
    backward_argmax = similarity.argmax(axis=0)
    mutual_fraction = float(np.mean([backward_argmax[target] == index for index, target in enumerate(forward_argmax)]))
    return {
        "rgb_patch_nearest_cosine_mean": float(forward.mean()),
        "rgb_patch_nearest_cosine_p10": percentile(forward, 10),
        "rgb_patch_mutual_nearest_fraction": mutual_fraction,
    }


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    """Return the symmetric, bounded divergence between two distributions."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / max(float(left.sum()), 1e-12)
    right = right / max(float(right.sum()), 1e-12)
    midpoint = 0.5 * (left + right)

    def kl_divergence(source: np.ndarray, target: np.ndarray) -> float:
        mask = source > 0
        return float(np.sum(source[mask] * np.log(source[mask] / np.maximum(target[mask], 1e-12))))

    return 0.5 * kl_divergence(left, midpoint) + 0.5 * kl_divergence(right, midpoint)


def pair_features(content_path: Path, reference_path: Path) -> dict[str, float]:
    content = fit_square_crop(content_path)
    reference = fit_square_crop(reference_path)
    content_edges = edge_features(content)
    reference_edges = edge_features(reference)
    content_appearance = appearance_features(content)
    reference_appearance = appearance_features(reference)
    output: dict[str, float] = {}
    for name, value in content_edges.items():
        if name.startswith("_"):
            continue
        output[f"content_{name}"] = value
    for name, value in reference_edges.items():
        if name.startswith("_"):
            continue
        output[f"reference_{name}"] = value
        output[f"reference_minus_content_{name}"] = value - content_edges[name]
        output[f"reference_div_content_{name}"] = value / max(content_edges[name], 1e-8)
    lab_mean = [content_appearance[f"lab_mean_{channel}"] - reference_appearance[f"lab_mean_{channel}"] for channel in "lab"]
    lab_std = [content_appearance[f"lab_std_{channel}"] - reference_appearance[f"lab_std_{channel}"] for channel in "lab"]
    output.update(
        {
            "lab_mean_distance": float(np.linalg.norm(lab_mean)),
            "lab_std_distance": float(np.linalg.norm(lab_std)),
            "contrast_abs_difference": abs(content_appearance["contrast_std"] - reference_appearance["contrast_std"]),
            "log_laplacian_variance_ratio": math.log((reference_appearance["laplacian_variance"] + 1e-6) / (content_appearance["laplacian_variance"] + 1e-6)),
            "high_frequency_abs_difference": abs(content_appearance["high_frequency_ratio"] - reference_appearance["high_frequency_ratio"]),
            "color_histogram_distance": float(np.linalg.norm(content_appearance["hsv_histogram"] - reference_appearance["hsv_histogram"])),
            "orientation_distribution_js": js_divergence(
                content_edges["_orientation_distribution"],
                reference_edges["_orientation_distribution"],
            ),
            **coarse_patch_features(content, reference),
        }
    )
    return output


def median(values: list[float]) -> float | str:
    return float(np.median(values)) if values else ""


def numeric(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if value.strip().upper() in {"", "NA", "N/A", "NONE"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def normalize_legacy_v22_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert the V2.2 canonical review schema to the V2.4 schema in memory."""
    normalized = []
    for row in rows:
        lam = numeric(row.get("lambda"))
        if lam is None:
            continue
        takeover = row.get("human_takeover_score_0_3", "")
        normalized.append(
            {
                "case": row["case"],
                "seed": row["seed"],
                "lambda": row["lambda"],
                "human_style_score_0_4": row.get("human_style_score_0_4", ""),
                "baseline_takeover_0_3": takeover if lam == 0.2 else "NA",
                "incremental_takeover_0_3": "NA" if lam == 0.2 else takeover,
                "style_valid": "true",
                "reference": row.get("human_reference_leakage_note", ""),
                "review_note": row.get("human_review_note", ""),
            }
        )
    return normalized


def aggregate_human(rows: list[dict[str, str]]) -> dict[str, object]:
    seeds = sorted({row["seed"] for row in rows})
    valid_rows = [row for row in rows if row.get("style_valid", "").lower() == "true"]
    baseline = [
        [
            score
            for row in rows
            if row["seed"] == seed
            and numeric(row.get("lambda")) == 0.2
            for score in [numeric(row.get("baseline_takeover_0_3"))]
            if score is not None
        ]
        for seed in seeds
    ]
    style_by_seed: dict[str, dict[float, float]] = {}
    incremental_by_seed: dict[str, dict[float, float]] = {}
    for seed in seeds:
        style_by_seed[seed] = {
            lam: score
            for row in valid_rows
            if row["seed"] == seed
            for lam in [numeric(row.get("lambda"))]
            for score in [numeric(row.get("human_style_score_0_4"))]
            if lam is not None and score is not None
        }
        incremental_by_seed[seed] = {
            lam: score
            for row in rows
            if row["seed"] == seed
            for lam in [numeric(row.get("lambda"))]
            for score in [numeric(row.get("incremental_takeover_0_3"))]
            if lam is not None and lam > 0.2 and score is not None
        }
    style_at_02 = [values[0.2] for values in style_by_seed.values() if 0.2 in values]
    style_at_10 = [values[1.0] for values in style_by_seed.values() if 1.0 in values]
    style_gain = [values[1.0] - values[0.2] for values in style_by_seed.values() if 0.2 in values and 1.0 in values]
    incremental_max = [max(values.values()) for values in incremental_by_seed.values() if values]
    interval_counts = [sum(value > 0 for value in values.values()) for values in incremental_by_seed.values()]
    late = [any(value > 0 for lam, value in values.items() if lam >= 0.8) for values in incremental_by_seed.values()]
    return {
        "seed_count": len(seeds),
        "baseline_takeover_median": median([value for values in baseline for value in values]),
        "baseline_takeover_max": max((value for values in baseline for value in values), default=""),
        "style_at_02_median": median(style_at_02),
        "style_at_10_median": median(style_at_10),
        "style_gain_median": median(style_gain),
        "incremental_takeover_max_median": median(incremental_max),
        "incremental_nonzero_interval_count": median(interval_counts),
        "late_escalation_frequency": float(np.mean(late)) if late else "",
        "style_valid_rate": len(valid_rows) / len(rows) if rows else 0.0,
        "label_status": "complete" if len(seeds) >= 3 else "seed42_only",
    }


def profile_label(row: dict[str, object]) -> str:
    initial = row["baseline_takeover_median"]
    style_at_10 = row["style_at_10_median"]
    if not isinstance(initial, (float, int)):
        return "pending"
    if initial >= 2:
        return "P4_high_initial_susceptibility"
    if isinstance(style_at_10, (float, int)) and style_at_10 >= 3:
        return "P1_low_risk_high_response" if initial <= 0 else "P3_moderate_susceptibility_high_response"
    return "P2_low_response"


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
    xr = average_ranks(x)
    yr = average_ranks(y)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return ""
    return float(np.corrcoef(xr, yr)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="analysis")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    if not V23_CASES.exists():
        raise FileNotFoundError(f"V2.3 pair manifest 不存在：{V23_CASES}")
    v23_cases = read_csv(V23_CASES)
    v15_cases = read_csv(V15_CASES)
    cases = []
    for source in v15_cases:
        if source["case_id"] in {"v1_5_demuth_church", "v1_5_kulhanek_snow_winter", "v1_5_demuth_wave"}:
            cases.append({"case_id": source["case_id"], "content_path": source["content"], "style_path": source["style"]})
    cases.extend({"case_id": row["canonical_case_id"], "content_path": row["content_path"], "style_path": row["style_path"]} for row in v23_cases)
    if len(cases) != 13 or len({case["case_id"] for case in cases}) != 13:
        raise ValueError(f"预期 13 个唯一 pair，实际得到 {len(cases)}：{cases}")
    human_rows = read_csv(V23_HUMAN) if V23_HUMAN.exists() else []
    single_seed_rows = read_csv(V23_HUMAN_SEED42) if V23_HUMAN_SEED42.exists() else []
    human_by_case: dict[str, list[dict[str, str]]] = {}
    for row in human_rows:
        human_by_case.setdefault(row["case"], []).append(row)
    multi_seed_cases = set(human_by_case)
    for row in single_seed_rows:
        if row["case"] not in multi_seed_cases:
            human_by_case.setdefault(row["case"], []).append(row)
    legacy_rows = read_csv(V22_CANONICAL_HUMAN) if V22_CANONICAL_HUMAN.exists() else []
    for row in normalize_legacy_v22_rows(legacy_rows):
        human_by_case.setdefault(row["case"], []).append(row)
    profile_rows = []
    feature_rows = []
    for case in cases:
        case_id = case["case_id"]
        labels = aggregate_human(human_by_case[case_id]) if case_id in human_by_case else {
            "seed_count": 0,
            "baseline_takeover_median": "",
            "baseline_takeover_max": "",
            "style_at_02_median": "",
            "style_at_10_median": "",
            "style_gain_median": "",
            "incremental_takeover_max_median": "",
            "incremental_nonzero_interval_count": "",
            "late_escalation_frequency": "",
            "style_valid_rate": "",
            "label_status": "pending_v2_4_profile_label",
        }
        labels["profile_label"] = profile_label(labels)
        content_path = ROOT / case["content_path"]
        reference_path = ROOT / case["style_path"]
        missing = [str(path) for path in (content_path, reference_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{case_id} 的输入图不存在：{missing}")
        feature = pair_features(content_path, reference_path)
        profile_rows.append({"case": case_id, **labels})
        feature_rows.append({"case": case_id, "content_path": case["content_path"], "reference_path": case["style_path"], "feature_source": "512px fit_square_crop; RGB patch fallback; CLIP/DINO unavailable locally", **feature})
    write_csv(output_dir / "v2_4_pair_profiles.csv", profile_rows)
    write_csv(output_dir / "v2_4_pair_features.csv", feature_rows)

    label_fields = ["baseline_takeover_median", "style_gain_median", "incremental_takeover_max_median"]
    feature_fields = [field for field in feature_rows[0] if field not in {"case", "content_path", "reference_path", "feature_source"} and not field.endswith("_histogram")]
    correlations = []
    for label_field in label_fields:
        for feature_field in feature_fields:
            values = [(float(profile[label_field]), float(feature[feature_field])) for profile, feature in zip(profile_rows, feature_rows) if str(profile[label_field]).strip() and profile["label_status"] in {"complete", "seed42_only"}]
            correlations.append({"label": label_field, "feature": feature_field, "n": len(values), "spearman_rho": spearman([pair[1] for pair in values], [pair[0] for pair in values])})
    write_csv(output_dir / "v2_4_feature_correlations.csv", correlations)
    scored_profiles = [row for row in profile_rows if row["label_status"] in {"complete", "seed42_only"}]
    feature_by_case = {row["case"]: row for row in feature_rows}
    checks = []

    def add_group_check(name: str, members_a: list[dict[str, object]], members_b: list[dict[str, object]]) -> None:
        for feature_field in feature_fields:
            values_a = [float(feature_by_case[row["case"]][feature_field]) for row in members_a]
            values_b = [float(feature_by_case[row["case"]][feature_field]) for row in members_b]
            if not values_a or not values_b:
                continue
            median_a = float(np.median(values_a))
            median_b = float(np.median(values_b))
            checks.append({
                "comparison": name,
                "feature": feature_field,
                "n_a": len(values_a),
                "n_b": len(values_b),
                "median_a": median_a,
                "median_b": median_b,
                "median_b_minus_a": median_b - median_a,
            })

    low_initial = [row for row in scored_profiles if float(row["baseline_takeover_median"]) == 0]
    high_initial = [row for row in scored_profiles if float(row["baseline_takeover_median"]) >= 2]
    add_group_check("baseline_takeover_0_vs_ge2", low_initial, high_initial)

    demuth = [row for row in scored_profiles if row["case"] in {
        "clean_demuth_G1_water_lake",
        "clean_demuth_G1_forest",
        "v1_5_demuth_church",
        "clean_demuth_G4_city_mismatch",
    }]
    for feature_field in feature_fields:
        values = [float(feature_by_case[row["case"]][feature_field]) for row in demuth]
        if values:
            checks.append({
                "comparison": "demuth_content_subset_span",
                "feature": feature_field,
                "n_a": len(values),
                "n_b": 0,
                "median_a": float(np.median(values)),
                "median_b": "",
                "median_b_minus_a": float(max(values) - min(values)),
            })

    low_risk = [row for row in scored_profiles if float(row["baseline_takeover_median"]) <= 1]
    high_style = [row for row in low_risk if float(row["style_gain_median"]) >= 2]
    low_style = [row for row in low_risk if float(row["style_gain_median"]) <= 1]
    add_group_check("low_risk_high_style_vs_low_style", low_style, high_style)
    write_csv(output_dir / "v2_4_preflight_checks.csv", checks)
    (output_dir / "v2_4_feature_manifest.json").write_text(
        json.dumps({
            "image_generation": False,
            "learned_features": "pending_local_checkpoint",
            "pair_count": len(cases),
            "scored_pair_count": len(scored_profiles),
            "feature_count": len(feature_fields),
            "profile_label_count": len({row["profile_label"] for row in scored_profiles}),
            "canonical_pairs_pending_human_labels": [row["case"] for row in profile_rows if row["label_status"] == "pending_v2_4_profile_label"],
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
