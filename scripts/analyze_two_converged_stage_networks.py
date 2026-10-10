#!/usr/bin/env python3
"""Describe the two converged, fixed-stage GSE74243 DiSCGRN networks.

This is a read-only analysis of existing fitted networks. It does not fit a
model, select alpha, infer a change point, or use the unconverged third stage.
The UMAP points are bulk lung *biological samples*, not individual cells.
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

from ctcpgrn.io import read_gse74243_expression


STAGE_DIRS = {
    "prenatal": Path("results/global_three_stage_500_30tf_prenatal_active_strict_20261007"),
    "early_postnatal": Path("results/global_three_stage_500_30tf_early_active_strict_20261007"),
}
STAGE_LABELS = {"prenatal": "Prenatal", "early_postnatal": "Early postnatal"}
STAGE_COLORS = {"prenatal": "#3274A1", "early_postnatal": "#DC753F"}
STRAIN_MARKERS = {"AJ": "o", "B6": "s", "C3H": "^"}


def load_converged_edges(stage_dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    outputs = {}
    for stage, directory in stage_dirs.items():
        diagnostics = pd.read_csv(directory / "fit_diagnostics.tsv", sep="\t")
        row = diagnostics.loc[diagnostics.stage.eq(stage)]
        if len(row) != 1 or not bool(row.iloc[0].converged):
            raise ValueError(f"The {stage} fit has not passed its convergence check")
        edges = pd.read_csv(directory / "selected_stage_edges.tsv.gz", sep="\t")
        if len(edges) != int(row.iloc[0].n_selected_edges):
            raise ValueError(f"Edge count differs from diagnostics for {stage}")
        if edges.duplicated(["regulator", "target"]).any():
            raise ValueError(f"Duplicate directed edges for {stage}")
        outputs[stage] = edges
    return outputs


def load_sample_panel(expression_path: Path, candidate_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    with np.load(candidate_dir / "strain_time_expression.npz", allow_pickle=False) as archive:
        genes = archive["genes"].astype(str).tolist()
    ranking = pd.read_csv(candidate_dir / "stage_differential_gene_ranking.tsv", sep="\t")
    panel = ranking.set_index("Gene Symbol").loc[genes].reset_index()[["Gene ID", "Gene Symbol"]]
    expression, metadata = read_gse74243_expression(expression_path)
    metadata = metadata.loc[metadata.time.gt(9.5) & metadata.time.le(31.0)].copy().reset_index(drop=True)
    selected = expression.set_index("Gene ID").loc[panel["Gene ID"].tolist()]
    values = selected[metadata["sample"].tolist()].to_numpy(dtype=float).T
    if values.shape[1] != len(genes) or not np.isfinite(values).all():
        raise ValueError("Selected sample expression does not match the fitted gene panel")
    samples = pd.DataFrame(values, index=metadata["sample"], columns=genes)
    metadata["network_stage"] = np.where(metadata.time.lt(20.0), "prenatal", "early_postnatal")
    return samples, metadata, genes


def load_unsupervised_umap_matrix(expression_path: Path, metadata: pd.DataFrame,
                                  n_genes: int = 2000) -> pd.DataFrame:
    """Select variable genes without using stage labels or the fitted panel."""
    expression, _ = read_gse74243_expression(expression_path)
    sample_columns = metadata["sample"].tolist()
    values = expression[sample_columns].to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    variances = np.var(values[finite], axis=1, ddof=1)
    work = expression.loc[finite, ["Gene ID", "Gene Symbol", *sample_columns]].copy()
    work["variance"] = variances
    work = work.sort_values(["variance", "Gene ID"], ascending=[False, True],
                            kind="stable")
    work = work.drop_duplicates("Gene ID").head(n_genes)
    return pd.DataFrame(work[sample_columns].to_numpy(dtype=float).T,
                        index=sample_columns, columns=work["Gene ID"].astype(str))


def import_umap():
    """Disable unavailable Numba disk caching in the legacy DyGMamba environment."""
    import numba.core.dispatcher as dispatcher
    import numba.np.ufunc.ufuncbuilder as ufuncbuilder

    dispatcher.Dispatcher.enable_caching = lambda self: None
    ufuncbuilder.UFuncDispatcher.enable_caching = lambda self: None
    import umap

    return umap


def embed_samples(samples: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    scaled = StandardScaler().fit_transform(samples.to_numpy())
    principal = PCA(n_components=min(30, len(samples) - 1, samples.shape[1]), random_state=42).fit_transform(scaled)
    umap = import_umap()
    coordinates = umap.UMAP(n_neighbors=15, min_dist=0.25, metric="euclidean",
                            random_state=42, n_jobs=1).fit_transform(principal)
    embedding = metadata[["sample", "strain", "stage", "time", "network_stage"]].copy()
    embedding["UMAP1"] = coordinates[:, 0]
    embedding["UMAP2"] = coordinates[:, 1]
    return embedding


def choose_regulators(edges: dict[str, pd.DataFrame], candidate_dir: Path,
                      n_each: int = 2) -> list[tuple[str, str]]:
    """Choose regulators by the count of displayed non-regulator targets."""
    all_regulators = set(pd.read_csv(candidate_dir / "selected_regulators.tsv",
                                    sep="\t")["Gene Symbol"].astype(str))
    chosen = []
    for stage in STAGE_DIRS:
        visible = edges[stage].loc[~edges[stage].target.isin(all_regulators)]
        counts = visible.groupby("regulator").size().sort_values(ascending=False)
        for regulator in counts.head(n_each).index:
            chosen.append((stage, str(regulator)))
    return chosen


def calculate_regulon_proxy(samples: pd.DataFrame, metadata: pd.DataFrame,
                            edges: dict[str, pd.DataFrame],
                            chosen: list[tuple[str, str]],
                            candidate_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed signed-target expression scores; not biochemical TF activity.

    For a TF, use the signs of its top 20 fitted non-TF targets from its
    discovery stage. Apply this fixed signature to every individual sample,
    then average the scores by strain and displayed developmental stage.
    """
    scaled = pd.DataFrame(StandardScaler().fit_transform(samples),
                          index=samples.index, columns=samples.columns)
    all_regulators = set(pd.read_csv(candidate_dir / "selected_regulators.tsv",
                                    sep="\t")["Gene Symbol"].astype(str))
    signatures = []
    scores = []
    for stage, regulator in chosen:
        part = edges[stage].loc[edges[stage].regulator.eq(regulator)].copy()
        part = part.loc[~part.target.isin(all_regulators)]
        part = part.assign(abs_weight=part.edge_weight.abs()).sort_values(
            ["abs_weight", "target"], ascending=[False, True]).head(20)
        if part.empty:
            raise ValueError(f"No non-regulator targets for {regulator} in {stage}")
        signs = np.sign(part.edge_weight.to_numpy(dtype=float))
        profile = scaled[part.target.tolist()].to_numpy() @ signs / len(part)
        label = f"{regulator} ({'E' if stage == 'prenatal' else 'P'})"
        signatures.extend({"regulator": regulator, "discovery_stage": stage,
                           "target": row.target, "edge_weight": row.edge_weight,
                           "edge_sign": int(np.sign(row.edge_weight)),
                           "signature_size": len(part)}
                          for row in part.itertuples())
        scores.extend({"sample": sample, "strain": strain, "time": time,
                       "network_stage": network_stage, "regulator": regulator,
                       "discovery_stage": stage, "regulon_label": label,
                       "signed_target_expression_score": float(value)}
                      for sample, strain, time, network_stage, value in zip(
                          metadata["sample"], metadata.strain, metadata.time,
                          metadata.network_stage, profile))
    return pd.DataFrame(scores), pd.DataFrame(signatures)


def select_target_edges(edges: dict[str, pd.DataFrame],
                        chosen: list[tuple[str, str]],
                        candidate_dir: Path, n_each: int = 6) -> pd.DataFrame:
    all_regulators = set(pd.read_csv(candidate_dir / "selected_regulators.tsv",
                                    sep="\t")["Gene Symbol"].astype(str))
    output = []
    for stage, regulator in chosen:
        part = edges[stage].loc[edges[stage].regulator.eq(regulator)].copy()
        part = part.loc[~part.target.isin(all_regulators)]
        part = part.assign(abs_weight=part.edge_weight.abs()).sort_values(
            ["abs_weight", "target"], ascending=[False, True]).head(n_each)
        part["display_regulator"] = f"{regulator} ({'E' if stage == 'prenatal' else 'P'})"
        part["display_rank"] = np.arange(1, len(part) + 1)
        output.append(part)
    return pd.concat(output, ignore_index=True)


def save_figure(embedding: pd.DataFrame, activity: pd.DataFrame,
                target_edges: pd.DataFrame, chosen: list[tuple[str, str]],
                output_dir: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(13.6, 8.2), constrained_layout=True)
    layout = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.3], height_ratios=[1.0, 1.18])
    ax_map = fig.add_subplot(layout[0, 0])
    ax_activity = fig.add_subplot(layout[0, 1])
    ax_network = fig.add_subplot(layout[1, :])

    for stage in STAGE_DIRS:
        for strain, marker in STRAIN_MARKERS.items():
            subset = embedding.loc[embedding.network_stage.eq(stage) & embedding.strain.eq(strain)]
            ax_map.scatter(subset.UMAP1, subset.UMAP2, s=36, color=STAGE_COLORS[stage],
                           marker=marker, edgecolor="white", linewidth=0.45, alpha=0.85,
                           label=f"{STAGE_LABELS[stage]}, {strain}")
    ax_map.set(xlabel="UMAP 1", ylabel="UMAP 2", title="a  Unsupervised bulk-sample UMAP")
    ax_map.legend(frameon=False, fontsize=7, ncol=2, loc="best", handletextpad=0.3,
                  columnspacing=0.7)

    labels = [f"{regulator} ({'E' if stage == 'prenatal' else 'P'})"
              for stage, regulator in chosen]
    group_order = [(stage, strain) for stage in STAGE_DIRS for strain in STRAIN_MARKERS]
    mean_activity = activity.groupby(["regulon_label", "network_stage", "strain"],
                                     observed=True)["signed_target_expression_score"].mean()
    matrix = np.array([[mean_activity.loc[(label, stage, strain)]
                        for stage, strain in group_order] for label in labels])
    activity_limit = max(0.15, np.quantile(np.abs(matrix), 0.95))
    image = ax_activity.imshow(matrix, aspect="auto", cmap="RdBu_r",
                               norm=TwoSlopeNorm(vmin=-activity_limit, vcenter=0,
                                                 vmax=activity_limit))
    ax_activity.set_xticks(range(len(group_order)),
                           [f"{'E' if stage == 'prenatal' else 'P'} · {strain}"
                            for stage, strain in group_order], rotation=0)
    ax_activity.set_yticks(range(len(labels)), labels)
    ax_activity.set_title("b  Regulon activity proxy from signed target expression")
    ax_activity.axvline(2.5, color="white", lw=2.5)
    fig.colorbar(image, ax=ax_activity, fraction=0.033, pad=0.02,
                 label="Mean signed target z-score")

    target_rows = list(dict.fromkeys(target_edges.target.astype(str).tolist()))
    values = np.full((len(target_rows), len(labels)), np.nan)
    row_index = {gene: i for i, gene in enumerate(target_rows)}
    column_index = {label: j for j, label in enumerate(labels)}
    for row in target_edges.itertuples():
        values[row_index[row.target], column_index[row.display_regulator]] = row.edge_weight
    finite = values[np.isfinite(values)]
    limit = max(0.1, np.quantile(np.abs(finite), 0.98))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#F0F1F3")
    heat = ax_network.imshow(values, aspect="auto", cmap=cmap,
                              norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    ax_network.set_xticks(range(len(labels)), labels)
    ax_network.set_yticks(range(len(target_rows)), target_rows, fontsize=8)
    ax_network.set_title("c  Strongest inferred regulator → target coefficients by stage")
    ax_network.tick_params(length=0)
    for i in range(len(target_rows)):
        for j in range(len(labels)):
            if np.isfinite(values[i, j]):
                ax_network.text(j, i, f"{values[i, j]:+.2f}", ha="center", va="center",
                                color="white" if abs(values[i, j]) > limit * 0.55 else "#222222",
                                fontsize=7)
    fig.colorbar(heat, ax=ax_network, fraction=0.017, pad=0.01,
                 label="ODE coefficient (signed)")
    fig.suptitle("DiSCGRN | two converged lung developmental networks", fontsize=14,
                 fontweight="bold")
    fig.savefig(output_dir / "two_stage_network_case.png", dpi=280, bbox_inches="tight")
    fig.savefig(output_dir / "two_stage_network_case.pdf", bbox_inches="tight")
    plt.close(fig)


def save_target_expression_heatmap(samples: pd.DataFrame, metadata: pd.DataFrame,
                                   target_edges: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Display the actual expression of the example targets in six sample groups."""
    genes = list(dict.fromkeys(target_edges.target.astype(str)))
    scaled = pd.DataFrame(StandardScaler().fit_transform(samples[genes]),
                          index=samples.index, columns=genes)
    groups = [(stage, strain) for stage in STAGE_DIRS for strain in STRAIN_MARKERS]
    means = np.column_stack([
        scaled.loc[metadata.loc[metadata.network_stage.eq(stage)
                                 & metadata.strain.eq(strain), "sample"], genes]
        .mean(axis=0).to_numpy()
        for stage, strain in groups
    ])
    result = pd.DataFrame(means, index=genes,
                          columns=[f"{stage}_{strain}" for stage, strain in groups])
    result.index.name = "target"
    result.to_csv(output_dir / "top_target_group_mean_expression_z.tsv", sep="\t")
    limit = max(0.5, np.quantile(np.abs(means), 0.98))
    fig, ax = plt.subplots(figsize=(7.3, 8.7), constrained_layout=True)
    image = ax.imshow(means, aspect="auto", cmap="RdBu_r",
                      norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    ax.set_xticks(range(6), [f"{'E' if stage == 'prenatal' else 'P'} · {strain}"
                              for stage, strain in groups])
    ax.set_yticks(range(len(genes)), genes, fontsize=8)
    ax.axvline(2.5, color="white", lw=2.5)
    ax.set_title("Expression of displayed target genes across stages and strains")
    ax.tick_params(length=0)
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025,
                 label="Mean expression z-score")
    fig.savefig(output_dir / "top_target_expression_heatmap.png", dpi=280,
                bbox_inches="tight")
    fig.savefig(output_dir / "top_target_expression_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path,
                        default=Path("data/processed/three_stage_global_500_30tf_20261007"))
    parser.add_argument("--expression", type=Path,
                        default=Path("data/raw/gse74243/GSE74243_matrix_RMAlog2expressionvalues.txt.gz"))
    parser.add_argument("--prenatal-dir", type=Path, default=STAGE_DIRS["prenatal"])
    parser.add_argument("--early-dir", type=Path, default=STAGE_DIRS["early_postnatal"])
    parser.add_argument("--embedding-tsv", type=Path,
                        help="Reuse an existing unsupervised embedding of exactly the same biological samples")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/real_two_stage_network_case_v3_20261007"))
    args = parser.parse_args()
    stage_dirs = {"prenatal": args.prenatal_dir, "early_postnatal": args.early_dir}
    edges = load_converged_edges(stage_dirs)
    samples, metadata, genes = load_sample_panel(args.expression, args.candidate_dir)
    umap_samples = load_unsupervised_umap_matrix(args.expression, metadata)
    chosen = choose_regulators(edges, args.candidate_dir)
    activity, signatures = calculate_regulon_proxy(samples, metadata, edges, chosen,
                                                    args.candidate_dir)
    target_edges = select_target_edges(edges, chosen, args.candidate_dir)
    if args.embedding_tsv is None:
        embedding = embed_samples(umap_samples, metadata)
    else:
        embedding = pd.read_csv(args.embedding_tsv, sep="\t")
        if len(embedding) != len(metadata) or set(embedding["sample"]) != set(metadata["sample"]):
            raise ValueError("Existing embedding does not cover exactly the current biological samples")
        embedding = embedding.set_index("sample").loc[metadata["sample"]].reset_index()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to replace existing results: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    embedding.to_csv(args.output_dir / "sample_umap.tsv", sep="\t", index=False)
    activity.to_csv(args.output_dir / "sample_regulon_expression_scores.tsv.gz",
                    sep="\t", index=False, compression="gzip")
    signatures.to_csv(args.output_dir / "regulon_signatures.tsv", sep="\t", index=False)
    target_edges.to_csv(args.output_dir / "top_target_edges.tsv", sep="\t", index=False)
    all_edges = pd.concat(edges.values(), ignore_index=True)
    all_edges.to_csv(args.output_dir / "both_converged_stage_edges.tsv.gz",
                     sep="\t", index=False, compression="gzip")
    rows = []
    for stage, part in edges.items():
        rows.append({"stage": stage, "n_edges": len(part),
                     "n_regulators_with_edges": part.regulator.nunique(),
                     "n_targets_with_edges": part.target.nunique(),
                     "n_positive": int(part.edge_weight.gt(0).sum()),
                     "n_negative": int(part.edge_weight.lt(0).sum())})
    pd.DataFrame(rows).to_csv(args.output_dir / "stage_network_summary.tsv", sep="\t", index=False)
    regulator_summary = pd.concat([
        part.groupby("regulator").edge_weight.agg(n_targets="size",
                                                  sum_abs_weight=lambda x: x.abs().sum())
        .reset_index().assign(stage=stage) for stage, part in edges.items()], ignore_index=True)
    regulator_summary.to_csv(args.output_dir / "regulator_network_summary.tsv",
                             sep="\t", index=False)
    save_figure(embedding, activity, target_edges, chosen, args.output_dir)
    save_target_expression_heatmap(samples, metadata, target_edges, args.output_dir)
    prenatal_edges = set(zip(edges["prenatal"].regulator, edges["prenatal"].target))
    postnatal_edges = set(zip(edges["early_postnatal"].regulator,
                              edges["early_postnatal"].target))
    candidates = pd.read_csv(args.candidate_dir / "candidate_tf_target_pairs.tsv.gz", sep="\t")
    prenatal_candidates = candidates.loc[candidates.stage.eq("prenatal")]
    postnatal_candidates = candidates.loc[candidates.stage.eq("early_postnatal")]
    candidate_sets = [set(zip(part.regulator, part.target))
                      for part in (prenatal_candidates, postnatal_candidates)]
    candidate_overlap = len(candidate_sets[0] & candidate_sets[1])
    common_path = args.prenatal_dir.parent.parent / "common_candidate_edges.tsv.gz"
    if common_path.exists():
        candidate_overlap = len(pd.read_csv(common_path, sep="\t"))
    mean_scores = activity.groupby(["regulon_label", "network_stage"])[
        "signed_target_expression_score"].mean().unstack()
    summary = {
        "analysis": "Exploratory analysis of two existing fixed-stage converged networks; no model refit",
        "sample_level": "Individual bulk lung biological samples, not single cells",
        "n_samples_in_embedding": len(metadata),
        "umap_variable_genes": umap_samples.shape[1],
        "umap_feature_selection": "Top-variance measured probe IDs among the displayed samples, without stage labels or fitted GRN gene panel",
        "n_genes_in_panel": len(genes),
        "stage_windows": {"prenatal": "E11.5 to P0 (P0 is a shared model boundary)",
                          "early_postnatal": "P0 to P11"},
        "umap_display_assignment": "Prenatal if time < P0; early postnatal if P0 <= time <= P11",
        "chosen_regulators_rule": "Top two regulators by selected non-regulator-target edge count within each converged stage",
        "chosen_regulators": [{"stage": stage, "regulator": tf} for stage, tf in chosen],
        "selected_directed_edge_overlap": len(prenatal_edges & postnatal_edges),
        "candidate_directed_edge_overlap": candidate_overlap,
        "mean_regulon_proxy_by_stage": mean_scores.round(4).to_dict(orient="index"),
        "activity_definition": "Mean signed z-scored expression of up to 20 strongest non-TF targets, with each signature fixed from its discovery-stage network; this is an expression-based proxy, not measured TF activity",
        "target_heatmap_rule": "Top six non-TF targets by absolute fitted coefficient for each chosen regulator",
        "expression_heatmap": "The same example targets, z-scored across displayed biological samples and averaged by stage and strain",
        "caveats": ["The stage boundaries were fixed before fitting and not inferred by this analysis",
                    "UMAP and activity use bulk expression and cannot establish cell-intrinsic regulation",
                    "The fitted target panel was screened for stage differential expression; the activity proxy is descriptive, not independent evidence of stage differences",
                    "The unconverged late postnatal network is excluded"],
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
