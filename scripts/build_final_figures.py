#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "final_figures"
OUT.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(ROOT / "results/iteration_refinement/validation_summary.tsv", sep="\t").set_index("method")
methods = ["ct", "euler", "dplsq", "two_stage"]
labels = ["CT-CP-GRN", "Euler-DP", "DPLSQ-DP", "Two-stage"]
colors = ["#17365D", "#4F81BD", "#9BBB59", "#A5A5A5"]
metrics = [
    ("network_aupr", "Network AUPR", True),
    ("change_aupr", "Changed-edge AUPR", True),
    ("cp_f1", "Change-point F1", True),
    ("rmse", "One-step RMSE", False),
]
fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2))
for ax, (column, title, _) in zip(axes.flat, metrics):
    values = [summary.loc[m, column] for m in methods]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title, fontweight="bold")
    ax.tick_params(axis="x", rotation=22, labelsize=8)
    ax.grid(axis="y", alpha=.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(values)*1.22)
fig.tight_layout()
fig.savefig(OUT / "final_method_comparison.png", dpi=240, bbox_inches="tight")
plt.close(fig)

blocked = pd.read_csv(ROOT / "results/final_validation/blocked_cv.tsv", sep="\t")
order = ["ct", "euler", "dplsq"]
means = blocked.groupby("method").rmse.mean().reindex(order)
stds = blocked.groupby("method").rmse.std().reindex(order)
fig, ax = plt.subplots(figsize=(7.6, 4.2))
bars = ax.bar(["CT-CP-GRN", "Euler-DP", "DPLSQ-DP"], means, yerr=stds, capsize=5, color=colors[:3])
ax.set_ylabel("Blocked CV RMSE")
ax.set_title("Time-blocked validation over 5 independent systems", fontweight="bold")
ax.grid(axis="y", alpha=.2)
for bar, value in zip(bars, means):
    ax.text(bar.get_x()+bar.get_width()/2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "blocked_cv.png", dpi=240, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.6, 3.9))
categories = ["Weak case\nseed 91001", "Detected case\nseed 96001"]
exact = [0, 1]
nochange = [1, 0]
x = np.arange(2)
ax.bar(x, exact, color="#17365D", label="Exact change point")
ax.bar(x, nochange, bottom=exact, color="#C9C9C9", label="No change selected")
ax.set_xticks(x, categories)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Fraction of 200 bootstraps")
ax.legend(frameon=False, ncol=2, loc="upper center")
ax.set_title("Bootstrap sensitivity to signal strength", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "bootstrap_sensitivity.png", dpi=240, bbox_inches="tight")
plt.close(fig)
