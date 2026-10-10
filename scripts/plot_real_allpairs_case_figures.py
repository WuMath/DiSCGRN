#!/usr/bin/env python3
"""Replot the converged fixed-P0, no-correlation-screen real-data case.

All figures use saved outputs from the all-TF-target coupled ODE fit. No
network coefficients, UMAP coordinates, or enrichment tests are recomputed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "results/real_p0_all_tf_targets_stage_case_20261008"
REGULATORS = ROOT / "results/real_p0_all_tf_targets_all_regulators_20261008"
PATHWAYS = ROOT / "results/real_p0_all_tf_targets_pathways_20261008"
OUTPUT = ROOT / "results/real_p0_top20_specific_tf_figures_20261008"
COMPOSITE_OUTPUT = ROOT / "results/real_p0_composite_panels_20261008"
STAGES = ("prenatal", "early_postnatal")
STAGE_LABEL = {"prenatal": "Embryonic E11.5–P0", "early_postnatal": "Early postnatal P0–P11"}
STAGE_COLOR = {"prenatal": "#2474A6", "early_postnatal": "#DE7140"}
STRAIN_MARKER = {"AJ": "o", "B6": "s", "C3H": "^"}


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_umap(case: Path, out: Path, panel_only: bool = False) -> dict:
    data = pd.read_csv(case / "sample_umap.tsv", sep="\t")
    if len(data) != 139 or set(data.network_stage) != set(STAGES):
        raise ValueError("UMAP input does not match the 139-sample fixed-P0 case")
    if panel_only:
        fig, ax = plt.subplots(figsize=(6.0, 4.9), constrained_layout=True)
        for stage in STAGES:
            part = data.loc[data.network_stage.eq(stage)]
            ax.scatter(part.UMAP1, part.UMAP2, c=STAGE_COLOR[stage],
                       marker="o", s=36, linewidths=0.3,
                       edgecolors="white", alpha=0.83, label=STAGE_LABEL[stage])
        ax.legend(loc="best", frameon=False, fontsize=11)
        ax.set(xlabel="UMAP 1", ylabel="UMAP 2")
        ax.tick_params(labelsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, out, "panel_a_stage_umap")

        fig, ax = plt.subplots(figsize=(6.0, 4.9), constrained_layout=True)
        strain_colors = {"AJ": "#4A6680", "B6": "#6F9982", "C3H": "#A97576"}
        for strain, marker in STRAIN_MARKER.items():
            part = data.loc[data.strain.eq(strain)]
            ax.scatter(part.UMAP1, part.UMAP2, c=strain_colors[strain],
                       marker=marker, s=36, linewidths=0.3,
                       edgecolors="white", alpha=0.83, label=strain)
        ax.legend(loc="best", frameon=False, fontsize=11)
        ax.set(xlabel="UMAP 1", ylabel="UMAP 2")
        ax.tick_params(labelsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, out, "panel_b_strain_umap")
        return {"samples": len(data), "strain_counts": data.strain.value_counts().to_dict(),
                "panels": 2}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
    for stage in STAGES:
        for strain, marker in STRAIN_MARKER.items():
            part = data.loc[data.network_stage.eq(stage) & data.strain.eq(strain)]
            axes[0].scatter(part.UMAP1, part.UMAP2, c=STAGE_COLOR[stage],
                            marker=marker, s=40, linewidths=0.35,
                            edgecolors="white", alpha=0.83)
    axes[0].legend(handles=[Line2D([0], [0], marker="o", linestyle="", color="white",
                                   markerfacecolor=STAGE_COLOR[s], markersize=8,
                                   label=STAGE_LABEL[s]) for s in STAGES],
                   loc="upper center", bbox_to_anchor=(0.50, 1.0),
                   frameon=False, fontsize=9)
    strain_colors = {"AJ": "#4A6680", "B6": "#6F9982", "C3H": "#A97576"}
    for strain, marker in STRAIN_MARKER.items():
        part = data.loc[data.strain.eq(strain)]
        axes[1].scatter(part.UMAP1, part.UMAP2, c=strain_colors[strain],
                        marker=marker, s=40, linewidths=0.35,
                        edgecolors="white", alpha=0.83, label=strain)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.50, 1.0),
                   frameon=False, fontsize=9)
    for ax, title in zip(axes, ("a  Developmental period", "b  Mouse strain")):
        ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title=title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Bulk lung samples | unsupervised expression UMAP", fontsize=13)
    save(fig, out, "bulk_sample_umap")
    return {"samples": len(data), "strain_counts": data.strain.value_counts().to_dict()}


def plot_bubbles(regulators: Path, case: Path, out: Path,
                 panel_only: bool = False) -> dict:
    coverage = pd.read_csv(regulators / "candidate_regulator_coverage.tsv", sep="\t")
    edges = pd.read_csv(case / "both_converged_stage_edges.tsv.gz", sep="\t")
    expected_edges = int(coverage.prenatal_all_edges.sum()
                         + coverage.early_postnatal_all_edges.sum())
    if len(coverage) != 30 or len(edges) != expected_edges:
        raise ValueError("Regulator coverage or stage edges do not match the converged run")
    rows = []
    for _, item in coverage.iterrows():
        for stage in STAGES:
            part = edges.loc[edges.stage.eq(stage) & edges.regulator.eq(item.regulator)]
            rows.append({"regulator": item.regulator, "stage": stage,
                         "selected_targets": part.target.nunique(),
                         "sum_absolute_weight": float(part.edge_weight.abs().sum()),
                         "mean_absolute_weight": float(part.edge_weight.abs().mean()) if len(part) else np.nan})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "regulator_selected_target_bubbles.tsv", sep="\t", index=False)
    order = coverage.sort_values(["total_edges", "regulator"], ascending=[False, True]).regulator.head(20).tolist()
    summary.loc[summary.regulator.isin(order)].to_csv(
        out / "top20_regulator_selected_target_bubbles.tsv", sep="\t", index=False)
    y = {tf: len(order) - i - 1 for i, tf in enumerate(order)}
    fig, ax = plt.subplots(figsize=(7.1, 7.9) if panel_only else (8.2, 7.9),
                           constrained_layout=True)
    displayed = summary.loc[summary.regulator.isin(order)]
    max_mean = 0.5
    for x, stage in enumerate(STAGES):
        part = displayed.loc[displayed.stage.eq(stage)]
        ax.scatter(np.full(len(part), x), part.regulator.map(y),
                   s=24 + 28 * np.sqrt(part.selected_targets.to_numpy()),
                   c=part.mean_absolute_weight.fillna(0), cmap="viridis",
                   vmin=0, vmax=max_mean, edgecolors="#263443", linewidths=0.65)
        for item in part.itertuples():
            ax.annotate(str(item.selected_targets), (x, y[item.regulator]),
                        xytext=(14, 0), textcoords="offset points", va="center",
                        fontsize=13 if panel_only else 7.5)
    highlighted = {"Rb1", "Zeb1", "Zbtb20", "Ezh2"}
    if not highlighted.issubset(order):
        raise ValueError("The four displayed case regulators must occur among the top 20")
    for tf in highlighted:
        ax.axhspan(y[tf] - 0.43, y[tf] + 0.43, color="#EAF1F3", zorder=-1)
    ax.set_yticks(range(len(order)), order[::-1], fontsize=16 if panel_only else 8)
    xlabels = ("E11.5–P0", "P0–P11") if panel_only else (
        STAGE_LABEL["prenatal"], STAGE_LABEL["early_postnatal"])
    ax.set_xticks((0, 1), xlabels)
    ax.set_xlim(-0.4, 1.45)
    ax.set_ylim(-0.65, len(order) - 0.35)
    if not panel_only:
        ax.set_title("Top 20 regulators by selected target count", fontsize=13)
    ax.grid(axis="y", color="#E4E8EB", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="both", length=0,
                   labelsize=13 if panel_only else 8)
    if not panel_only:
        ax.text(0.01, -0.06, "Bubble area ∝ √(selected target count); number = selected targets",
                transform=ax.transAxes, fontsize=8)
    norm = plt.Normalize(0, max_mean)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax,
                        pad=0.02, fraction=0.035)
    cbar.set_label("Mean |ODE coefficient|", fontsize=13 if panel_only else 9)
    if panel_only:
        cbar.ax.tick_params(labelsize=12)
    save(fig, out, "panel_c_regulator_bubbles" if panel_only else "top20_selected_target_bubbles")
    return {"regulators_screened": len(coverage), "regulators_displayed": len(order),
            "stage_edge_counts": summary.groupby("stage").selected_targets.sum().to_dict(),
            "highlighted_regulators": sorted(highlighted)}


def plot_weight_heatmap(case: Path, regulators: Path, out: Path,
                        panel_only: bool = False) -> dict:
    edges = pd.read_csv(case / "both_converged_stage_edges.tsv.gz", sep="\t")
    coverage = pd.read_csv(regulators / "candidate_regulator_coverage.tsv", sep="\t")
    if len(coverage) != 30:
        raise ValueError("Expected 30 preselected candidate regulators")
    top20 = set(coverage.sort_values(["total_edges", "regulator"],
                                     ascending=[False, True]).head(20).regulator)
    prenatal = coverage.sort_values(["prenatal_all_edges", "regulator"],
                                    ascending=[False, True]).head(2).regulator.tolist()
    postnatal_top = coverage.sort_values(["early_postnatal_all_edges", "regulator"],
                                         ascending=[False, True]).iloc[0].regulator
    postnatal_exclusive = coverage.loc[coverage.regulator.ne(postnatal_top)
                                        & coverage.prenatal_all_edges.eq(0)
                                        & coverage.early_postnatal_all_edges.ge(2)]
    postnatal_exclusive = postnatal_exclusive.sort_values(
        ["early_postnatal_all_edges", "regulator"], ascending=[False, True])
    if postnatal_exclusive.empty:
        raise ValueError("No early-postnatal-exclusive regulator with at least two selected targets")
    selected_regulators = [("prenatal", tf) for tf in prenatal]
    selected_regulators += [("early_postnatal", postnatal_top),
                            ("early_postnatal", postnatal_exclusive.iloc[0].regulator)]
    if len(set(selected_regulators)) != 4 or any(tf not in top20 for _, tf in selected_regulators):
        raise ValueError("Case regulators must be four distinct members of the bubble-plot top 20")
    cmap = plt.colormaps["RdBu_r"].copy()
    cmap.set_bad("#E9EDF0")
    rows = []
    for primary_stage, tf in selected_regulators:
        part = edges.loc[edges.stage.eq(primary_stage) & edges.regulator.eq(tf)].copy()
        part["absolute"] = part.edge_weight.abs()
        part = part.sort_values(["absolute", "target"], ascending=[False, True]).head(10)
        if part.empty:
            raise ValueError(f"No fitted edges for {tf} in {primary_stage}")
        for rank, item in enumerate(part.itertuples(), start=1):
            rows.append({"stage": primary_stage, "regulator": tf,
                         "target": item.target, "rank_within_regulator": rank,
                         "edge_weight": float(item.edge_weight)})
    selected = pd.DataFrame(rows)
    selected.to_csv(out / "selected_tf_top10_target_weights.tsv", sep="\t", index=False)
    targets = selected.target.drop_duplicates().tolist()
    matrix = np.full((len(targets), len(selected_regulators)), np.nan)
    for item in selected.itertuples():
        matrix[targets.index(item.target), selected_regulators.index((item.stage, item.regulator))] = item.edge_weight
    scale = max(float(np.nanmax(np.abs(matrix))), 1e-6)
    linthresh = max(scale / 100, 1e-6)
    norm = SymLogNorm(linthresh=linthresh, linscale=0.9,
                      vmin=-scale, vmax=scale)
    fig, ax = plt.subplots(figsize=(8.4, max(7.0, 0.29 * len(targets) + 2.2)),
                           constrained_layout=True)
    ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm,
              interpolation="none", aspect="auto")
    ax.set_yticks(np.arange(len(targets)), targets,
                  fontsize=16 if panel_only else 8)
    ax.set_xticks(np.arange(len(selected_regulators)),
                  [f"{tf}\n({'E' if stage == 'prenatal' else 'P'})" if panel_only
                   else f"{tf}\n{'Embryonic' if stage == 'prenatal' else 'Early postnatal'}"
                   for stage, tf in selected_regulators], fontsize=9)
    ax.tick_params(axis="x", labelsize=14 if panel_only else 9)
    ax.set_ylabel("Target gene", fontsize=15 if panel_only else 10)
    ax.tick_params(length=0)
    ax.axvline(1.5, color="#35404A", linewidth=1.2)
    for i, j in np.argwhere(np.isfinite(matrix)):
        value = matrix[i, j]
        ax.text(j, i, f"{value:+.2g}", ha="center", va="center",
                fontsize=11 if panel_only else 6.5,
                color="white" if abs(value) > scale / 4 else "#20272D")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        shrink=0.72, pad=0.035)
    cbar.set_label("Signed ODE coefficient (symmetric-log colour scale)",
                   fontsize=13 if panel_only else 10)
    if panel_only:
        cbar.ax.tick_params(labelsize=12)
    if not panel_only:
        fig.suptitle("Top fitted targets of stage-enriched regulators",
                     fontsize=13)
    save(fig, out, "panel_d_target_weights" if panel_only else "selected_tf_top10_target_weights")
    counts = {f"{stage}:{tf}": int((selected.regulator.eq(tf)
                                  & selected.stage.eq(stage)).sum())
              for stage, tf in selected_regulators}
    return {"selection_rule": "Top two embryonic regulators by selected targets; "
            "top early-postnatal regulator plus top early-postnatal-only regulator "
            "with at least two targets, all within bubble-plot top 20",
            "selected_regulators": [f"{stage}:{tf}" for stage, tf in selected_regulators],
            "displayed_targets_per_regulator": counts,
            "unique_targets": len(targets),
            "unshown_cells": int(np.isnan(matrix).sum()),
            "largest_absolute_coefficient": scale}


def plot_enrichment(pathways: Path, out: Path, panel_only: bool = False) -> dict:
    go = pd.read_csv(pathways / "GO_BP_significant.tsv", sep="\t")
    group = go.loc[go.group.eq("early_postnatal_Zbtb20_top50")].sort_values("q_value")
    significant = not group.empty
    if group.empty:
        all_tests = pd.read_csv(pathways / "GO_BP_all_tests.tsv.gz", sep="\t")
        group = all_tests.loc[all_tests.group.eq(
            "early_postnatal_Zbtb20_top50")].sort_values("q_value")
    if group.empty:
        raise ValueError("No GO tests for Zbtb20 top-50 targets")
    # Suppress near-duplicate GO descendants/ancestors with almost identical
    # overlap sets. This affects display only; q values remain those of all tests.
    chosen = []
    overlap_sets = []
    for item in group.itertuples():
        genes = set(str(item.overlap_genes).split(","))
        if all(len(genes & previous) / len(genes | previous) < 0.70 for previous in overlap_sets):
            chosen.append(item)
            overlap_sets.append(genes)
        if len(chosen) == 7:
            break
    display = pd.DataFrame(chosen).iloc[::-1].copy()
    display.to_csv(out / "displayed_go_terms.tsv", sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    y = np.arange(len(display))
    score = -np.log10(display.q_value.to_numpy())
    sizes = 30 + 15 * display.overlap_count.to_numpy()
    scat = ax.scatter(score, y, s=sizes, c=display.fold_enrichment,
                      cmap="magma_r", vmin=2, vmax=10, edgecolors="#383A40", linewidths=0.5)
    if panel_only:
        shortened = {
            "microtubule-based process": "microtubule process",
            "regulation of cell cycle": "cell-cycle regulation",
            "positive regulation of G2/M transition of mitotic cell cycle": "G2/M transition",
            "regulation of mitotic cell cycle phase transition": "mitotic phase transition",
        }
        labels = [shortened.get(term, term) for term in display.term_name]
    else:
        labels = display.term_name.tolist()
    ax.set_yticks(y, labels, fontsize=14 if panel_only else 9)
    ax.set_xlabel("−log₁₀(BH-adjusted q)", fontsize=13 if panel_only else 10)
    if not panel_only:
        ax.set_title("GO biological processes | Zbtb20 top-50 fitted targets"
                     + ("" if significant else " (none pass FDR 0.05)"), fontsize=12)
    ax.axvline(-np.log10(0.05), linestyle="--", color="#74808A", linewidth=1)
    ax.grid(axis="x", color="#E5E8EA", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if panel_only:
        ax.tick_params(axis="x", labelsize=12)
    cbar = fig.colorbar(scat, ax=ax, pad=0.02)
    cbar.set_label("Fold enrichment", fontsize=13 if panel_only else 10)
    if panel_only:
        cbar.ax.tick_params(labelsize=12)
    if not panel_only:
        ax.text(0, -0.17, "Dot area reflects overlap size; 485 connectable non-TF genes form the background.",
                transform=ax.transAxes, fontsize=8)
    save(fig, out, "panel_f_zbtb20_go_enrichment" if panel_only else "go_enrichment_zbtb20_top50")
    kegg = pd.read_csv(pathways / "KEGG_significant.tsv", sep="\t")
    return {"go_significant_terms_for_set": int((group.q_value < 0.05).sum()),
            "go_plot_is_descriptive": not significant,
            "displayed_go_terms": len(display),
            "kegg_significant_terms": len(kegg), "top_go_q": float(group.q_value.min())}


def plot_rb1_enrichment(pathways: Path, out: Path, panel_only: bool = False) -> dict:
    go = pd.read_csv(pathways / "GO_BP_all_tests.tsv.gz", sep="\t")
    group = go.loc[go.group.eq("prenatal_Rb1_top50")].sort_values(
        ["p_value", "term_id"])
    if group.empty:
        raise ValueError("No GO tests for prenatal Rb1 top-50 targets")
    candidate = group.loc[group.p_value.lt(0.05) & group.overlap_count.ge(3)]
    chosen, overlap_sets = [], []
    for item in candidate.itertuples():
        genes = set(str(item.overlap_genes).split(","))
        if all(len(genes & previous) / len(genes | previous) < 0.70
               for previous in overlap_sets):
            chosen.append(item)
            overlap_sets.append(genes)
        if len(chosen) == 7:
            break
    if not chosen:
        raise ValueError("No descriptive Rb1 GO terms meeting the display rule")
    display = pd.DataFrame(chosen).iloc[::-1].copy()
    display.to_csv(out / "displayed_rb1_go_terms.tsv", sep="\t", index=False)
    short_names = {
        "regulation of protein ubiquitination": "protein ubiquitination regulation",
        "regulation of post-translational protein modification": "post-translational modification",
        "regulation of protein modification by small protein conjugation or removal": "small-protein conjugation",
        "positive regulation of monoatomic ion transmembrane transport": "ion transport regulation",
    }
    labels = [short_names.get(name, name) for name in display.term_name]
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    y = np.arange(len(display))
    scores = -np.log10(display.p_value.to_numpy())
    scatter = ax.scatter(scores, y, s=30 + 15 * display.overlap_count.to_numpy(),
                         c=display.fold_enrichment, cmap="magma_r",
                         edgecolors="#383A40", linewidths=0.5)
    ax.set_yticks(y, labels, fontsize=14 if panel_only else 9)
    ax.set_xlabel("−log₁₀(nominal p)", fontsize=13 if panel_only else 10)
    if not panel_only:
        ax.set_title("Exploratory GO terms | embryonic Rb1 top-50 targets", fontsize=12)
    ax.axvline(-np.log10(0.05), linestyle="--", color="#74808A", linewidth=1)
    ax.grid(axis="x", color="#E5E8EA", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if panel_only:
        ax.tick_params(axis="x", labelsize=12)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Fold enrichment", fontsize=13 if panel_only else 10)
    if panel_only:
        colorbar.ax.tick_params(labelsize=12)
    save(fig, out, "panel_e_rb1_go_exploratory" if panel_only else "go_enrichment_rb1_top50_exploratory")
    kegg = pd.read_csv(pathways / "KEGG_all_tests.tsv.gz", sep="\t")
    rb1_kegg = kegg.loc[kegg.group.eq("prenatal_Rb1_top50")]
    return {"query_size": 50, "go_tests": len(group),
            "go_nominal_p_below_0_05": int(group.p_value.lt(0.05).sum()),
            "go_fdr_q_below_0_05": int(group.q_value.lt(0.05).sum()),
            "best_go_q": float(group.q_value.min()),
            "kegg_tests": len(rb1_kegg),
            "kegg_fdr_q_below_0_05": int(rb1_kegg.q_value.lt(0.05).sum()),
            "displayed_go_terms": len(display)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=CASE)
    parser.add_argument("--regulators-dir", type=Path, default=REGULATORS)
    parser.add_argument("--pathways-dir", type=Path, default=PATHWAYS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--composite-panels", action="store_true",
                        help="Export six title-free, single-panel PDFs for LaTeX composition")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = COMPOSITE_OUTPUT if args.composite_panels else OUTPUT
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "umap": plot_umap(args.case_dir, args.output_dir, args.composite_panels),
        "bubbles": plot_bubbles(args.regulators_dir, args.case_dir, args.output_dir,
                                args.composite_panels),
        "weights": plot_weight_heatmap(args.case_dir, args.regulators_dir,
                                       args.output_dir, args.composite_panels),
        "rb1_enrichment": plot_rb1_enrichment(args.pathways_dir, args.output_dir,
                                              args.composite_panels),
        "enrichment": plot_enrichment(args.pathways_dir, args.output_dir,
                                      args.composite_panels),
        "composite_panels": args.composite_panels,
        "interpretation": "Bulk-sample visualization and fitted effective edges; GO describes the selected target set, not causal validation.",
    }
    (args.output_dir / "figure_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
