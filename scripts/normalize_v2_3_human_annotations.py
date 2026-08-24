"""Normalize the completed V2.3 human review table into the fixed schema."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVALID_STYLE_CASES = {"compat_G4_city_mismatch"}
FIELDS = (
    "case",
    "seed",
    "lambda",
    "human_style_score_0_4",
    "baseline_takeover_0_3",
    "incremental_takeover_0_3",
    "style_valid",
    "reference",
    "review_note",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open(newline="", encoding=encoding) as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("gb18030", b"", 0, 1, f"Unable to decode {path}")


def normalize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if rows and set(FIELDS).issubset(rows[0]):
        return rows
    output = []
    for row in rows:
        lam = float(row["lambda"])
        style_valid = "false" if row["case"] in INVALID_STYLE_CASES else "true"
        style = row.get("human_style_score_0_4", "")
        if style_valid == "false":
            style = "NA"
        review_note = row.get("review_note", "") or row.get("human_review_note", "")
        if review_note.startswith(("当前 λ 的 seed 内绝对强度", "当前 λ 相比前一个 λ 新增异常")):
            review_note = ""
        if row.get("note"):
            review_note = "; ".join(value for value in (review_note, row["note"]) if value)
        output.append(
            {
                "case": row["case"],
                "seed": row["seed"],
                "lambda": row["lambda"],
                "human_style_score_0_4": style,
                "baseline_takeover_0_3": row.get("human_takeover_score_0_3", "") if lam == 0.2 else "NA",
                "incremental_takeover_0_3": "NA" if lam == 0.2 else row.get("human_takeover_score_0_3", ""),
                "style_valid": style_valid,
                "reference": row.get("reference", ""),
                "review_note": review_note,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="runs/ip_adapter_plus_injection/v2_3_pair_response_profiles/audits/human_sensitivity_annotations.csv",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    input_path = ROOT / args.input
    output_path = ROOT / (args.output or args.input)
    rows = normalize(read_rows(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)


if __name__ == "__main__":
    main()
