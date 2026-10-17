from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results" / "tables" / "time_matched_summary.csv"
OUTPUT_DIR = ROOT / "results" / "figures"
ORDER = [
    "FlashST-Full",
    "Random-FT",
    "CoreSet-FT",
    "Proxy-FT",
    "ELFS-FT",
    "BADGE-FT",
    "UHerding-FT",
    "PAD-TF",
]
LABELS = {
    "FlashST-Full": "FlashST-Full",
    "Random-FT": "Random-FT",
    "CoreSet-FT": "CoreSet-FT",
    "Proxy-FT": "Proxy-FT",
    "ELFS-FT": "ELFS-FT",
    "BADGE-FT": "BADGE-FT",
    "UHerding-FT": "UHerding-FT",
    "PAD-TF": "PAD-TF (Ours)",
}


def main() -> None:
    with SUMMARY_PATH.open(encoding="utf-8") as stream:
        records = {row["method"]: row for row in csv.DictReader(stream)}
    missing = [method for method in ORDER if method not in records]
    if missing:
        raise RuntimeError(f"Missing time-matched summaries: {missing}")
    values = np.asarray(
        [100.0 * float(records[method]["relative_rmse_reduction_mean"]) for method in ORDER]
    )
    errors = np.asarray(
        [100.0 * float(records[method]["relative_rmse_reduction_ci95"]) for method in ORDER]
    )
    positions = np.arange(len(ORDER))
    colors = ["#B8C5D6"] * len(ORDER)
    colors[-1] = "#D94B4B"
    best_index = int(np.argmax(values))
    if best_index != len(ORDER) - 1:
        colors[best_index] = "#315A8A"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    figure, axis = plt.subplots(figsize=(3.35, 3.05))
    axis.barh(
        positions,
        values,
        xerr=errors,
        color=colors,
        edgecolor="#34495E",
        linewidth=0.55,
        error_kw={"elinewidth": 0.8, "capsize": 2.0, "capthick": 0.8},
    )
    axis.axvline(values[-1], color="#D94B4B", linestyle="--", linewidth=0.9, alpha=0.85)
    axis.set_yticks(positions, [LABELS[method] for method in ORDER])
    axis.invert_yaxis()
    axis.set_xlabel("Relative RMSE reduction (%)")
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.55, linestyle="--")
    axis.set_axisbelow(True)
    lower = max(0.0, float(np.min(values - errors)) - 4.0)
    upper = float(np.max(values + errors)) + 4.5
    axis.set_xlim(lower, upper)
    for position, value in zip(positions, values):
        axis.text(value + 0.35, position, f"{value:.1f}", va="center", ha="left", fontsize=7.5)
    axis.text(
        0.99,
        1.015,
        "Per-pair time cap = PAD-TF total time",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#555555",
    )
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout(pad=0.35)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "figure3_time_matched.png"
    pdf_path = OUTPUT_DIR / "figure3_time_matched.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print(f"best_method={ORDER[best_index]}")
    print(f"png={png_path}")
    print(f"pdf={pdf_path}")


if __name__ == "__main__":
    main()
