#!/usr/bin/env python3
"""Create publication figures for the mandatory revision experiments."""

import matplotlib.pyplot as plt
import pandas as pd
from ctcpgrn.experiment_params import load_frozen_params
from ctcpgrn.paths import results_root

OUT = results_root() / "revision_figures"


def e1():
    path = results_root() / "revision_e1/e1_summary.tsv"
    if not path.exists():
        return
    d = pd.read_csv(path, sep="\t")
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for irr, g in d.groupby("irregularity"):
        g = g.sort_values("kappa_mean")
        ax.errorbar(g.kappa_mean, g.delta_network_aupr,
                    yerr=[g.delta_network_aupr-g.ci_low, g.ci_high-g.delta_network_aupr],
                    marker="o", capsize=3, label=f"irregularity={irr:g}")
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set(xlabel=r"$\kappa=\rho(A)\,\overline{\Delta t}$",
           ylabel="Network AUPR difference (DiSCGRN − Euler)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "e1_kappa_scan.pdf")
    fig.savefig(OUT / "e1_kappa_scan.png", dpi=300)


def e3():
    d = pd.read_csv(results_root() / "revision_e3/regularisation_path_15systems_summary.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.semilogx(d.alpha, d.network_aupr_mean, marker="o", label="DiSCGRN")
    ax.fill_between(d.alpha, d.network_aupr_mean-d.network_aupr_sd,
                    d.network_aupr_mean+d.network_aupr_sd, alpha=.15)
    ax.set(xlabel=r"Sparsity penalty $\alpha$", ylabel="Network AUPR")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "e3_regularisation_path.pdf")
    fig.savefig(OUT / "e3_regularisation_path.png", dpi=300)


def e4():
    path = results_root() / "revision_e4_single/e4_power_summary.tsv"
    if not path.exists():
        return
    d = pd.read_csv(path, sep="\t")
    d["strength"] = d.rewiring_fraction.astype(str) + "/" + d.weight_scale.astype(str)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    for ax, (T, g) in zip(axes, d.groupby("T")):
        order = (g[["rewiring_fraction", "weight_scale", "strength"]]
                 .drop_duplicates().sort_values(["rewiring_fraction", "weight_scale"]))
        for method, mg in g.groupby("method"):
            values = order.merge(mg, on=["rewiring_fraction", "weight_scale", "strength"])
            display_name = "DiSCGRN" if method.lower() in {"ct", "ctcpgrn"} else method.upper()
            ax.plot(range(len(values)), values.detection_rate, marker="o", label=display_name)
        ax.set_title(f"T={T}")
        ax.set_xticks(range(len(order)), order.strength, rotation=55, ha="right", fontsize=7)
        ax.set_xlabel("rewiring / weight scale")
    axes[0].set_ylabel("Change detection rate")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "e4_power.pdf")
    fig.savefig(OUT / "e4_power.png", dpi=300)


def e6():
    d = pd.read_csv(results_root() / "revision_e6/e6_summary.tsv", sep="\t")
    order = ["DiSCGRN", "Euler--DP", "EDISON", "VARDetect"]
    colours = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
    for ax, (T, group) in zip(axes, d.groupby("T")):
        group = group.set_index("method").reindex(order)
        bars = ax.bar(range(len(order)), group.network_aupr, color=colours)
        ax.set_title(f"T={T}")
        ax.set_xticks(range(len(order)), ["DiSCGRN", "Euler", "EDISON", "VARDetect"], rotation=25)
        ax.set_ylim(0, 0.65)
        for bar, value, evaluable in zip(bars, group.network_aupr, group.network_evaluable):
            ax.text(bar.get_x() + bar.get_width()/2, value + .015,
                    f"{value:.3f}\n({evaluable:.0%})", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("Stage-network AUPR")
    fig.tight_layout()
    fig.savefig(OUT / "e6_external_baselines.pdf")
    fig.savefig(OUT / "e6_external_baselines.png", dpi=300)


if __name__ == "__main__":
    load_frozen_params()
    OUT.mkdir(parents=True, exist_ok=True)
    e1(); e3(); e4(); e6()
