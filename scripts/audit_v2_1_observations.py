"""Audit directly observable regional differences in V2.1 generated outputs."""

from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/ip_adapter_plus_injection/v2_1_regional_pilot"
CASES = {
    "Church": "v1_5_demuth_church",
    "Snow": "v1_5_kulhanek_snow_winter",
    "Wave": "v1_5_demuth_wave",
}
SEEDS = (42, 123, 777)


def image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def mae(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    delta = np.abs(left - right).mean(axis=2)
    return float(delta[mask].mean())


def audit(case: str, seed: int) -> dict[str, float | int | str]:
    run_dir = RUN_ROOT / CASES[case] / f"seed{seed}"
    content = image(run_dir / "content.png")
    uniform = image(run_dir / "U.png")
    subject = image(run_dir / "S_subject.png")
    background = image(run_dir / "S_background.png")
    valid = image(run_dir / "masks/valid_eval.png")[:, :, 0] > 0
    subject_mask = image(run_dir / "masks/subject.png")[:, :, 0] > 0
    background_mask = image(run_dir / "masks/background.png")[:, :, 0] > 0

    values: dict[str, float | int | str] = {"case": case, "seed": seed}
    regions = {"global": valid, "subject": subject_mask, "background": background_mask}
    for name, mask in regions.items():
        values[f"B_minus_U_{name}"] = mae(background, uniform, mask)
        values[f"S_minus_U_{name}"] = mae(subject, uniform, mask)
        values[f"U_minus_content_{name}"] = mae(uniform, content, mask)
        values[f"S_minus_content_{name}"] = mae(subject, content, mask)
        values[f"B_minus_content_{name}"] = mae(background, content, mask)
    return values


def main() -> None:
    rows = [audit(case, seed) for case in CASES for seed in SEEDS]
    keys = (
        "B_minus_U_global",
        "S_minus_U_global",
        "B_minus_U_subject",
        "B_minus_U_background",
        "S_minus_U_subject",
        "S_minus_U_background",
        "U_minus_content_background",
        "S_minus_content_background",
        "B_minus_content_background",
        "U_minus_content_subject",
        "S_minus_content_subject",
        "B_minus_content_subject",
    )
    print("case seed " + " ".join(f"{key:>24}" for key in keys))
    for row in rows:
        print(
            f"{row['case']:5} {row['seed']:4} "
            + " ".join(f"{float(row[key]):24.3f}" for key in keys)
        )
    print("\nmean +/- sample std by case")
    for case in CASES:
        case_rows = [row for row in rows if row["case"] == case]
        print(case)
        for key in keys:
            values = np.asarray([float(row[key]) for row in case_rows])
            print(f"  {key:28}: {values.mean():.3f} +/- {values.std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
