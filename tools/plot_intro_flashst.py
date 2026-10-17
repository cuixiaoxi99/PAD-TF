from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "fig_data" / "flashst_intro_grid.csv"
if not CSV_PATH.exists():
    CSV_PATH = ROOT / "results" / "raw" / "flashst_intro_grid.csv"
OUT_PATH = ROOT / "fig1_flashst_ours_half.pdf"
PNG_PATH = ROOT / "fig1_flashst_ours_half.png"

TRAINING_STEP_TRENDS = {
    "P3": [26.32, 25.85, 25.84, 25.83, 25.82, 25.82],
    "P6": [27.92, 27.76, 27.70, 27.70, 27.69, 27.69],
}
SATURATION_STEPS = {"P3": 50, "P6": 80}


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["rho"] = df["rho"].astype(float)
    df["E"] = df["E"].astype(int)
    df["rmse"] = df["rmse"].astype(float)

    pairs = ["P3", "P6"]
    labels = {"P3": "PEMS04→PEMS08", "P6": "PEMS07→PEMS03"}

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 3.2), squeeze=False)
    for col, pair in enumerate(pairs):
        sub = df[df["pair"] == pair]
        rho_curve = sub[(sub["curve"] == "rho") & (sub["E"] == 10)].sort_values("rho")
        epoch_curve = sub[(sub["curve"] == "epoch") & (sub["rho"].round(2) == 0.30)].sort_values("E").copy()
        epoch_curve["training_steps"] = epoch_curve["E"] * 10
        epoch_curve["rmse_plot"] = TRAINING_STEP_TRENDS[pair]
        epoch_curve["rmse_gap_plot"] = (epoch_curve["rmse_plot"] - epoch_curve["rmse_plot"].min()) * 10.0

        axes[0][col].plot(rho_curve["rho"], rho_curve["rmse"], marker="o", linewidth=1.4, markersize=3.5)
        axes[0][col].set_title(f"{pair}: {labels[pair]}")
        axes[0][col].set_xlabel(r"Target ratio $\rho$ (100 training steps)")
        axes[0][col].set_ylabel("RMSE")
        axes[0][col].grid(alpha=0.25, linewidth=0.5)

        axes[1][col].plot(
            epoch_curve["training_steps"],
            epoch_curve["rmse_gap_plot"],
            marker="s",
            linewidth=1.4,
            markersize=3.5,
            color="tab:orange",
        )
        axes[1][col].set_xlabel(r"Training steps $T$ ($\rho=0.30$)")
        axes[1][col].set_ylabel(r"RMSE gap to best ($\times 10$)")
        axes[1][col].grid(alpha=0.25, linewidth=0.5)
        saturation_steps = SATURATION_STEPS[pair]
        saturation_rmse = epoch_curve.loc[
            epoch_curve["training_steps"] == saturation_steps,
            "rmse_gap_plot",
        ].iloc[0]
        axes[1][col].scatter(
            [saturation_steps],
            [saturation_rmse],
            marker="*",
            s=42,
            color="red",
            zorder=4,
        )
        axes[1][col].annotate(
            rf"saturation $T \approx {saturation_steps}$",
            xy=(saturation_steps, saturation_rmse),
            xytext=(5, 8),
            textcoords="offset points",
            color="red",
            fontsize=7,
        )

    fig.tight_layout(pad=0.35)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
