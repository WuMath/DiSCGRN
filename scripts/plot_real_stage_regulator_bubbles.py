"""Plot all candidate targets of the 30 regulators in the two P0 stages.

Targets include both regulators and non-regulators. Candidate pairs are
counted whether or not the subsequent sparse model gives them nonzero edge
weights. Colour encodes the screening correlation, not fitted ODE strength.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT / "data/processed/three_stage_global_500_30tf_20261007/candidate_tf_target_pairs.tsv.gz"
DEFAULT_REGULATORS = ROOT / "data/processed/three_stage_global_500_30tf_20261007/selected_regulators.tsv"
DEFAULT_OUTPUT = ROOT / "results/real_p0_rho015_stage_case_v2_20261007"
STAGES = ("prenatal", "early_postnatal")
STAGE_NAMES = ("Embryonic\nE11.5–P0", "Early postnatal\nP0–P11")


def make_summary(candidate_file: Path, regulator_file: Path) -> pd.DataFrame:
    regulators = pd.read_csv(regulator_file, sep="\t")["Gene Symbol"].astype(str).tolist()
    candidates = pd.read_csv(candidate_file, sep="\t")
    rows = []
    for stage in STAGES:
        stage_pairs = candidates.loc[candidates.stage.eq(stage)]
        counts = stage_pairs.groupby("regulator", sort=False).agg(
            n_candidate_targets=("target", "nunique"),
            mean_absolute_correlation=("within_strain_centered_correlation", lambda s: float(np.abs(s).mean())),
        )
        for regulator in regulators:
            entry = counts.loc[regulator]
            rows.append(
                {
                    "stage": stage,
                    "regulator": regulator,
                    "n_candidate_targets": int(entry["n_candidate_targets"]),
                    "mean_absolute_correlation": float(entry["mean_absolute_correlation"]),
                }
            )
    summary = pd.DataFrame(rows)
    summary["stage_rank"] = (
        summary.groupby("stage")["n_candidate_targets"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return summary


def make_plot(summary: pd.DataFrame, output: Path) -> None:
    order = (
        summary.loc[summary.stage.eq("prenatal")]
        .sort_values(["n_candidate_targets", "regulator"], ascending=[False, True])
        .regulator.tolist()
    )
    fig, ax = plt.subplots(figsize=(7.2, 9.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    y_pos = {name: len(order) - i - 1 for i, name in enumerate(order)}
    for name in ("Rb1", "Ruvbl1"):
        if name in y_pos:
            ax.axhspan(y_pos[name] - 0.44, y_pos[name] + 0.44, color="#eef5f0", zorder=0)
    for j, stage in enumerate(STAGES):
        subset = summary.loc[summary.stage.eq(stage)]
        x = np.full(len(subset), j, dtype=float)
        y = np.array([y_pos[name] for name in subset.regulator], dtype=float)
        n = subset.n_candidate_targets.to_numpy(dtype=int)
        scatter = ax.scatter(
            x,
            y,
            s=8 + 0.7 * n,
            c=subset.mean_absolute_correlation,
            vmin=0.3,
            vmax=0.85,
            cmap="viridis",
            edgecolors="#243b35",
            linewidths=0.6,
            zorder=3,
        )
        for x_i, y_i, n_i in zip(x, y, n):
            ax.annotate(str(n_i), (x_i, y_i), xytext=(17, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_xticks(range(len(STAGES)), STAGE_NAMES)
    ax.set_yticks(range(len(order)), order[::-1])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.65, len(order) - 0.35)
    ax.grid(axis="y", color="#e5e9e5", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="both", length=0, labelsize=8)
    ax.tick_params(axis="x", pad=9)
    for tick in ax.get_yticklabels():
        if tick.get_text() in ("Rb1", "Ruvbl1"):
            tick.set_fontweight("bold")
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.78, pad=0.035)
    cbar.set_label("Mean |screening correlation|", fontsize=9)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"regulator_stage_bubbles.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-file", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--regulators", type=Path, default=DEFAULT_REGULATORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = make_summary(args.candidate_file, args.regulators)
    assert len(summary) == 60 and summary.regulator.nunique() == 30
    summary.to_csv(args.output_dir / "regulator_bubble_summary.tsv", sep="\t", index=False)
    make_plot(summary, args.output_dir)
    print(summary.loc[summary.regulator.isin(["Rb1", "Ruvbl1", "Zbtb20"])].to_string(index=False))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
