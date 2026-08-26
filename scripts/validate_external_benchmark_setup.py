"""Validate the frozen external benchmark configuration without generating images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="external_benchmark/configs/external_eval_pairs_v1.csv")
    parser.add_argument("--prompts", default="external_benchmark/configs/prompts_v1.csv")
    parser.add_argument("--tracks", default="external_benchmark/configs/tracks.yaml")
    parser.add_argument("--methods", default="external_benchmark/configs/methods.yaml")
    parser.add_argument("--environments", default="external_benchmark/configs/environments.yaml")
    args = parser.parse_args()

    pairs = read_csv(ROOT / args.manifest)
    prompts = read_csv(ROOT / args.prompts)
    prompt_by_id = {row["pair_id"]: row for row in prompts}
    if len(pairs) != 24 or len({row["pair_id"] for row in pairs}) != 24:
        raise ValueError(f"Expected 24 unique held-out pairs, got {len(pairs)}")
    if set(prompt_by_id) != {row["pair_id"] for row in pairs}:
        raise ValueError("Pair manifest and prompt manifest do not have identical pair IDs")

    seen_content = set()
    seen_references = set()
    for row in pairs:
        content = ROOT / row["content_path"]
        reference = ROOT / row["reference_path"]
        if not content.exists() or not reference.exists():
            raise FileNotFoundError(f"Missing input for {row['pair_id']}: {content}, {reference}")
        with Image.open(content) as image:
            if image.width < 64 or image.height < 64:
                raise ValueError(f"Suspicious content dimensions: {row['pair_id']} {image.size}")
        with Image.open(reference) as image:
            if image.width < 64 or image.height < 64:
                raise ValueError(f"Suspicious reference dimensions: {row['pair_id']} {image.size}")
        seen_content.add(row["content_path"])
        seen_references.add(row["reference_path"])

    if len(seen_content) != 6 or len(seen_references) != 4:
        raise ValueError(f"Expected 6 content and 4 reference images, got {len(seen_content)} and {len(seen_references)}")
    if {row["content_family"] for row in pairs} != {"architecture", "urban", "flowers", "garden", "mountain_lake", "night_landscape"}:
        raise ValueError("Unexpected content-family coverage")
    if {row["reference_family"] for row in pairs} != {"monet", "van_gogh", "hokusai", "klimt"}:
        raise ValueError("Unexpected reference-family coverage")

    tracks = yaml.safe_load((ROOT / args.tracks).read_text(encoding="utf-8"))
    methods = yaml.safe_load((ROOT / args.methods).read_text(encoding="utf-8"))
    method_by_id = {row["method_id"]: row for row in methods["methods"]}
    declared_track_ids = {track["track_id"] for track in tracks["tracks"]}
    if len(declared_track_ids) != len(tracks["tracks"]):
        raise ValueError("Track manifest contains duplicate track IDs")
    for track in tracks["tracks"]:
        for method_id in track["method_ids"]:
            if method_id not in method_by_id:
                raise ValueError(f"Unknown method {method_id} in track {track['track_id']}")
            if track["track_id"] not in method_by_id[method_id].get("tracks", []):
                raise ValueError(f"Method {method_id} does not declare track {track['track_id']}")

    environments = yaml.safe_load((ROOT / args.environments).read_text(encoding="utf-8"))
    if environments.get("manager") != "uv":
        raise ValueError("External benchmark environments must use uv")
    environment_ids = {environment["environment_id"] for environment in environments["environments"]}
    if len(environment_ids) != len(environments["environments"]):
        raise ValueError("Environment manifest contains duplicate environment IDs")
    for environment in environments["environments"]:
        for requirement in environment.get("source_requirements", []):
            if not (ROOT / requirement).exists():
                raise FileNotFoundError(f"Missing environment requirement source: {requirement}")

    print(f"valid held-out pairs: {len(pairs)}")
    print(f"content images: {len(seen_content)}")
    print(f"reference images: {len(seen_references)}")
    print(f"tracks: {', '.join(sorted(declared_track_ids))}")
    print(f"environments: {', '.join(sorted(environment_ids))}")
    print("status: setup valid; no generation performed")


if __name__ == "__main__":
    main()
