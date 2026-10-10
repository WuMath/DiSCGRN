#!/usr/bin/env python3
"""Descriptive bulk-sample UMAP, NKX2-1 expression, and regulator-edge heatmaps.

The UMAP points are whole-lung expression samples, not cells. Marker groups in
the edge heatmap describe target genes; edge coefficients remain bulk-network
estimates by mouse strain and must not be called lineage-specific GRNs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

STRAINS = ("AJ", "B6", "C3H")
COLORS = {"AJ": "#0072B2", "B6": "#D55E00", "C3H": "#009E73"}
MARKERS = {"AJ": "o", "B6": "s", "C3H": "^"}
TARGET_GROUPS = {
    "Alveolar epithelial-associated": ("Ager", "Sftpb", "Sftpa1"),
    "Airway epithelial-associated": ("Scgb1a1", "Cyp2f2"),
    "Mesenchymal-associated": ("Dpt", "Igfbp7"),
    "Immune-associated": ("Lyz2", "H2-Ab1"),
}
ELIGIBLE_TFS = ("Nkx2-1", "Hif3a", "Nr1d1", "Nr4a1", "Epas1",
                "Meox2", "Snai2", "Foxp2", "Klf2")


def read_raw_panel(bundle: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(bundle / "sample_metadata.tsv", sep="\t")
    z = pd.read_csv(bundle / "sample_expression_zscore.tsv.gz", sep="\t")
    z = z.set_index("gene")
    normalisation = pd.read_csv(bundle / "normalisation.tsv", sep="\t")
    normalisation = normalisation.set_index(["strain", "gene"])
    if set(metadata["sample"]) != set(z.columns):
        raise ValueError("Expression samples do not match sample metadata")
    raw = z.loc[:, metadata["sample"]].T.copy()
    for strain in STRAINS:
        samples = metadata.loc[metadata.strain.eq(strain), "sample"]
        means = normalisation.loc[strain, "train_mean_log2_rma"].reindex(raw.columns)
        scales = normalisation.loc[strain, "train_sd_log2_rma"].reindex(raw.columns)
        if means.isna().any() or scales.isna().any():
            raise ValueError(f"Missing normalisation for {strain}")
        raw.loc[samples] = raw.loc[samples].to_numpy() * scales.to_numpy() + means.to_numpy()
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError("Non-finite reconstructed RMA expression")
    return metadata, raw


def sample_umap(metadata: pd.DataFrame, raw: pd.DataFrame, output: Path) -> pd.DataFrame:
    # Descriptive full-data embedding; never used for model selection or testing.
    scaled = StandardScaler().fit_transform(raw.to_numpy(float))
    pcs = PCA(n_components=15, random_state=42).fit_transform(scaled)
    embedding = umap.UMAP(n_neighbors=15, min_dist=0.2, metric="euclidean",
                          random_state=42, n_jobs=1).fit_transform(pcs)
    coords = metadata.copy()
    coords["UMAP1"], coords["UMAP2"] = embedding[:, 0], embedding[:, 1]
    coords["Nkx2-1_log2_RMA"] = raw["Nkx2-1"].to_numpy(float)
    coords.to_csv(output / "whole_lung_sample_umap_coordinates.tsv", sep="\t", index=False)

    summary = (coords.groupby(["strain", "stage", "time"])["Nkx2-1_log2_RMA"]
               .agg(["mean", "std", "count"]).reset_index()
               .sort_values(["strain", "time"]))
    summary.to_csv(output / "nkx21_expression_by_strain_time.tsv", sep="\t", index=False)

    plt.rcParams.update({"font.size": 10.5, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.4, 6.6), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.13])
    ax_time = fig.add_subplot(grid[0, 0])
    ax_strain = fig.add_subplot(grid[0, 1])
    ax_nkx = fig.add_subplot(grid[1, :])
    time_range = (float(coords.time.min()), float(coords.time.max()))
    for strain in STRAINS:
        sub = coords[coords.strain.eq(strain)]
        ax_time.scatter(sub.UMAP1, sub.UMAP2, c=sub.time, cmap="viridis",
                        vmin=time_range[0], vmax=time_range[1], s=28,
                        marker=MARKERS[strain], edgecolor="none", alpha=0.82)
        ax_strain.scatter(sub.UMAP1, sub.UMAP2, color=COLORS[strain], s=28,
                          marker=MARKERS[strain], edgecolor="none", alpha=0.82,
                          label=strain)
    scale = plt.cm.ScalarMappable(cmap="viridis",
                                  norm=plt.Normalize(vmin=time_range[0], vmax=time_range[1]))
    fig.colorbar(scale, ax=ax_time, shrink=0.8, label="Computational developmental day")
    ax_strain.legend(title="Mouse strain", frameon=False, loc="best", ncol=3,
                     fontsize=8, title_fontsize=8)
    for ax, title in ((ax_time, "a  Whole-lung samples by age"),
                      (ax_strain, "b  Same embedding by strain")):
        ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title=title)
        ax.spines[["top", "right"]].set_visible(False)
    for strain in STRAINS:
        sub = coords[coords.strain.eq(strain)]
        ax_nkx.scatter(sub.time, sub["Nkx2-1_log2_RMA"], color=COLORS[strain],
                       s=9, alpha=0.18, rasterized=True)
        agg = summary[summary.strain.eq(strain)]
        ax_nkx.plot(agg.time, agg["mean"], marker=MARKERS[strain], markersize=3,
                    lw=1.7, color=COLORS[strain], label=strain)
    ax_nkx.axvline(20, ls="--", lw=0.9, color="0.35")
    ax_nkx.set_xticks([11.5, 20, 27, 34, 50, 76],
                       ["E11.5", "P0", "P7", "P14", "P30", "P56"])
    ax_nkx.set(xlabel="Developmental age (P0 mapped to day 20)",
               ylabel="NKX2-1 expression (RMA log2)",
               title="c  NKX2-1 expression across developmental ages")
    ax_nkx.spines[["top", "right"]].set_visible(False)
    ax_nkx.legend(frameon=False, ncol=3)
    fig.savefig(output / "whole_lung_umap_nkx21_expression.pdf", bbox_inches="tight")
    plt.close(fig)
    return coords


def edge_heatmap(edges_path: Path, output: Path) -> dict:
    edges = pd.read_csv(edges_path, sep="\t")
    edges = edges[(edges.method == "discgrn") & (edges.stage == 1)]
    target_order = [gene for genes in TARGET_GROUPS.values() for gene in genes]
    missing = set(target_order) - set(edges.target)
    if missing:
        raise ValueError(f"Marker targets missing from fitted panel: {sorted(missing)}")
    ranking = (edges[edges.regulator.isin(ELIGIBLE_TFS)]
               .groupby("regulator").edge_weight.apply(lambda s: s.abs().sum())
               .sort_values(ascending=False))
    regulators = ranking.head(6).index.tolist()
    if "Nkx2-1" not in regulators:
        regulators[-1] = "Nkx2-1"
    rows = []
    matrices = {}
    for strain in STRAINS:
        subset = edges[edges.strain.eq(strain)]
        matrix = subset.pivot(index="regulator", columns="target", values="edge_weight")
        matrix = matrix.loc[regulators, target_order].to_numpy(float)
        matrices[strain] = matrix
        for r, regulator in enumerate(regulators):
            for c, target in enumerate(target_order):
                rows.append({"strain": strain, "regulator": regulator,
                             "target": target,
                             "target_group": next(k for k, v in TARGET_GROUPS.items() if target in v),
                             "edge_coefficient": matrix[r, c]})
    pd.DataFrame(rows).to_csv(output / "tf_marker_edge_weights_by_strain.tsv",
                              sep="\t", index=False)

    vmax = max(float(np.max(np.abs(m))) for m in matrices.values())
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.5), layout="constrained",
                             sharex=True, sharey=True)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    boundaries = np.cumsum([len(items) for items in TARGET_GROUPS.values()])[:-1]
    for ax, strain in zip(axes, STRAINS):
        matrix = matrices[strain]
        image = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_yticks(range(len(regulators)), regulators)
        ax.set_ylabel(f"{strain}\nregulator")
        ax.set_xticks(range(len(target_order)), target_order, rotation=45,
                      ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", labelbottom=True)
        for boundary in boundaries:
            ax.axvline(boundary - 0.5, color="0.25", lw=1.0)
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                value = matrix[r, c]
                if abs(value) >= 0.012:
                    ax.text(c, r, f"{value:+.2f}", ha="center", va="center",
                            fontsize=8.5, color="white" if abs(value) > 0.55 * vmax
                            else "black")
    start = 0
    for group, genes in TARGET_GROUPS.items():
        middle = start + (len(genes) - 1) / 2
        axes[0].text(middle, 1.12, group.replace("-associated", ""),
                     ha="center", va="bottom", fontsize=9.2,
                     transform=axes[0].get_xaxis_transform())
        start += len(genes)
    fig.colorbar(image, ax=axes, shrink=0.65,
                 label="Signed continuous-time edge coefficient")
    fig.savefig(output / "tf_marker_edge_heatmap_by_strain.pdf", bbox_inches="tight")
    plt.close(fig)
    return {"regulators": regulators, "targets_by_group": TARGET_GROUPS,
            "max_absolute_coefficient": vmax,
            "interpretation": "Bulk-tissue strain networks grouped by marker targets; not lineage-specific GRNs."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path,
                        default=Path("data/processed/real_study_20261004/model/full"))
    parser.add_argument("--edges", type=Path,
                        default=Path("results/real_data_20261006/full_network_edges.tsv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/real_data_20261006/sample_and_tf_visuals"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata, raw = read_raw_panel(args.bundle)
    coords = sample_umap(metadata, raw, args.output_dir)
    heatmap = edge_heatmap(args.edges, args.output_dir)
    (args.output_dir / "figure_settings.json").write_text(json.dumps({
        "n_samples": len(coords), "n_genes": raw.shape[1],
        "umap": {"pca_components": 15, "neighbors": 15, "min_dist": 0.2,
                 "seed": 42, "unit": "whole-lung biological sample"},
        "edge_heatmap": heatmap,
    }, indent=2) + "\n")
    print(f"Wrote descriptive figures and TSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
