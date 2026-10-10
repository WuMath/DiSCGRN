#!/usr/bin/env python3
"""Draw the three single-parameter sensitivity panels used by the paper.

The alpha/H curves share 15 systems with a fixed true boundary. The rho curve
uses a separate six-system calibration set and evaluates boundary selection.
Run from any directory: python scripts/plot_parameter_sensitivity_panels.py
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
E3 = ROOT / "results" / "revision_e3"
RHO = ROOT / "results" / "runs" / "weighted_score_20261004" / "weighted_score_calibration" / "rho_grid.tsv"
OUT = ROOT / "manuscript" / "figures"

BLUE = "#225ea8"
ORANGE = "#d95f0e"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.tick_params(labelsize=10)


def save(fig, name):
    fig.tight_layout(pad=0.8)
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def plot_alpha():
    d = pd.read_csv(E3 / "regularisation_path_15systems_summary.tsv", sep="\t").sort_values("alpha")
    assert len(d) == 12
    fig, ax = plt.subplots(figsize=(4.7, 2.7))
    ax.plot(d.alpha, d.network_aupr_mean, "o-", color=BLUE, lw=1.8, ms=4.2)
    ax.fill_between(d.alpha, d.network_aupr_mean - d.network_aupr_sd,
                    d.network_aupr_mean + d.network_aupr_sd, color=BLUE, alpha=0.12,
                    linewidth=0)
    ax.axvline(4e-4, color="#777777", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"Sparsity coefficient $\alpha$", fontsize=11)
    ax.set_ylabel("Stage-network AUPR", fontsize=11)
    ax.set_ylim(0.2, 0.7)
    style(ax)
    save(fig, "sensitivity_alpha.pdf")


def plot_budget():
    base = pd.read_csv(E3 / "ablation_15systems_summary.tsv", sep="\t")
    extra = pd.read_csv(E3 / "budget_extension_15systems_summary.tsv", sep="\t")
    d = pd.concat([base, extra], ignore_index=True)
    d = d[d.label.str.fullmatch(r"CT-a0004-EulerInit-H\d+")].copy()
    d["H"] = d.label.str.extract(r"-H(\d+)").astype(int)
    d = d.sort_values("H")
    assert d.H.tolist() == [40, 80, 160, 320, 640, 1000], d.H.tolist()
    fig, ax = plt.subplots(figsize=(4.7, 2.7))
    ax.plot(d.H, d.network_aupr_mean, "o-", color=BLUE, lw=1.8, ms=4.5)
    ax.axvline(80, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(d.H)
    ax.set_xticklabels([str(h) for h in d.H], rotation=30)
    ax.set_xlabel(r"Maximum iteration budget $H$", fontsize=11)
    ax.set_ylabel("Stage-network AUPR", fontsize=11)
    ax.set_ylim(0.515, 0.535)
    style(ax)
    save(fig, "sensitivity_budget.pdf")


def plot_rho():
    d = pd.read_csv(RHO, sep="\t")
    d = d[(d.method == "ct") & (d.alpha == 4e-4) & (d.max_iter == 80)]
    d = d.sort_values("beta_ratio")
    assert len(d) == 19
    fig, ax = plt.subplots(figsize=(4.7, 2.7))
    ax.plot(d.beta_ratio, d.mean_cp_f1, "o-", color=BLUE, lw=1.8, ms=3.5,
            label="Change-point matching")
    ax.plot(d.beta_ratio, d.null_specificity, "s-", color=ORANGE, lw=1.8,
            ms=3.5, label="No-change specificity")
    ax.axvspan(0.30, 0.45, color="#dddddd", alpha=0.55, linewidth=0)
    ax.axvline(0.30, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlim(-0.01, 0.91)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xticks([0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90])
    ax.set_xlabel(r"Change penalty ratio $\rho$", fontsize=11)
    ax.set_ylabel("Proportion of systems", fontsize=11)
    ax.legend(loc="center right", frameon=False, fontsize=9)
    style(ax)
    save(fig, "sensitivity_rho.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plot_alpha()
    plot_budget()
    plot_rho()
    print(f"Wrote three PDF panels to {OUT}")
