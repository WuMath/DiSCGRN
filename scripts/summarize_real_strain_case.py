#!/usr/bin/env python3
"""Summarize strain-specific NKX2-1 network candidates from existing fits.

This is a descriptive, post hoc case analysis. It reuses full-data networks and
the fixed-one-stage bootstrap fits; no model is retrained or tuned here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

STRAINS = ("AJ", "B6", "C3H")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    case_dir = args.results_dir / "case_study_nkx21"
    edges = pd.read_csv(args.results_dir / "full_network_edges.tsv", sep="\t")
    edges = edges[(edges.method == "discgrn") & (edges.stage == 1)].copy()
    edges["selected"] = edges.edge_weight.abs() > 1e-8
    all_edges = edges.pivot(index=["regulator", "target"], columns="strain", values="selected")
    selected_counts = {strain: int(all_edges[strain].sum()) for strain in STRAINS}
    overlap = {
        f"{left}_{right}": {
            "intersection": int((all_edges[left] & all_edges[right]).sum()),
            "union": int((all_edges[left] | all_edges[right]).sum()),
            "jaccard": float((all_edges[left] & all_edges[right]).sum()
                             / (all_edges[left] | all_edges[right]).sum()),
        }
        for left, right in (("AJ", "B6"), ("AJ", "C3H"), ("B6", "C3H"))
    }

    table = pd.read_csv(case_dir / "nkx21_case_all_targets.tsv", sep="\t")
    evidence = pd.read_csv(args.evidence, sep="\t")
    evidence = evidence[evidence.regulator.eq("Nkx2-1")]
    table = table.merge(evidence[["target", "collectri_reported"]], on="target",
                        how="left", validate="one_to_one")
    selected = {strain: table[f"weight_{strain}"].abs() > 1e-8 for strain in STRAINS}
    c3h_only = selected["C3H"] & ~selected["AJ"] & ~selected["B6"]
    # Frequency cutoffs are descriptive filters, not a significance test.
    stable = (c3h_only & table.frequency_C3H.ge(0.8)
              & table.frequency_AJ.le(0.2) & table.frequency_B6.le(0.2))
    focus = table.loc[stable].sort_values("weight_C3H", key=lambda x: x.abs(),
                                          ascending=False).copy()
    focus.to_csv(case_dir / "nkx21_stable_strain_candidates.tsv", sep="\t", index=False)

    summary = {
        "full_network_selected_edges": selected_counts,
        "pairwise_jaccard": overlap,
        "nkx21_selected_edges": {strain: int(selected[strain].sum()) for strain in STRAINS},
        "nkx21_C3H_only": int(c3h_only.sum()),
        "nkx21_C3H_only_lung_chip": int(table.loc[c3h_only, "chip_lung_support"].sum()),
        "nkx21_C3H_only_stable": int(stable.sum()),
        "nkx21_C3H_only_stable_lung_chip": int(focus.chip_lung_support.sum()),
        "nkx21_C3H_only_stable_collectri": int(focus.collectri_reported.sum()),
        "stable_candidate_targets": focus.target.tolist(),
        "notes": "Descriptive case selection after model fitting; not a biological significance test.",
    }
    (case_dir / "strain_network_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    target_sets = {
        strain: set(table.loc[selected[strain], "target"]) for strain in STRAINS
    }
    universe = set().union(*target_sets.values())
    regions = {
        code: sum(all((gene in target_sets[strain]) == (bit == "1")
                      for strain, bit in zip(STRAINS, code)) for gene in universe)
        for code in ("100", "010", "001", "110", "101", "011", "111")
    }
    if [len(target_sets[strain]) for strain in STRAINS] != [10, 10, 23]:
        raise ValueError("NKX2-1 target counts changed; recheck manuscript results")

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, (ax_overlap, ax_coeff) = plt.subplots(
        1, 2, figsize=(9.2, 3.9), layout="constrained",
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )
    circles = (
        ("AJ", (0.40, 0.51), "#2878B5"),
        ("B6", (0.65, 0.51), "#DE8037"),
        ("C3H", (0.525, 0.74), "#3A9D75"),
    )
    for name, center, color in circles:
        ax_overlap.add_patch(Circle(center, 0.32, facecolor=color, edgecolor=color,
                                    linewidth=1.7, alpha=0.23))
    positions = {
        "100": (0.20, 0.49), "010": (0.85, 0.49),
        "001": (0.525, 0.96), "110": (0.525, 0.30),
        "101": (0.31, 0.69), "011": (0.74, 0.69),
        "111": (0.525, 0.57),
    }
    for code, (x, y) in positions.items():
        ax_overlap.text(x, y, str(regions[code]), ha="center", va="center",
                        fontsize=13, fontweight="bold")
    for name, x, y in (("AJ (10)", 0.18, 0.12),
                       ("B6 (10)", 0.87, 0.12),
                       ("C3H (23)", 0.525, 1.13)):
        ax_overlap.text(x, y, name, ha="center", va="center", fontsize=10)
    ax_overlap.set(xlim=(0.05, 1.0), ylim=(0.05, 1.2), aspect="equal",
                   title="a  NKX2-1 target-set overlap")
    ax_overlap.axis("off")

    labels = [f"{row.target}{'*' if row.collectri_reported else ''}"
              for row in focus.itertuples()]
    weights = focus["weight_C3H"].to_numpy(float)
    frequencies = focus["frequency_C3H"].to_numpy(float)
    y = np.arange(len(focus))
    ax_coeff.barh(y, weights, height=0.62,
                  color=["#BF5B56" if value > 0 else "#4D86AA" for value in weights])
    ax_coeff.axvline(0, color="0.35", linewidth=0.8)
    ax_coeff.set_yticks(y, labels)
    ax_coeff.invert_yaxis()
    ax_coeff.set_xlim(-0.016, 0.064)
    ax_coeff.set_xlabel("C3H signed edge coefficient")
    ax_coeff.set_title("b  C3H-only stable candidate edges")
    for index, (weight, frequency) in enumerate(zip(weights, frequencies)):
        ax_coeff.text(0.062, index, f"{round(frequency * 30):d}/30",
                      ha="right", va="center", fontsize=8.5)
    ax_coeff.spines[["top", "right"]].set_visible(False)
    ax_coeff.grid(axis="x", color="0.88", linewidth=0.6)
    ax_coeff.set_axisbelow(True)
    output = case_dir / "nkx21_strain_network_case.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Figure: {output}")


if __name__ == "__main__":
    main()
