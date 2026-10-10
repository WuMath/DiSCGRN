#!/usr/bin/env python3
"""Prepare stage-ranked genes, regulators, and allowed TF-target pairs.

The ``all_tf_targets`` mode imposes no TF-target correlation screen. The
legacy ``screened`` mode is retained only to reproduce earlier analyses.
For predictive evaluation, rerun within each outer training fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f

from ctcpgrn.external import read_collectri
from ctcpgrn.io import read_gse74243_expression


STAGES = (("prenatal", 11.5, 20.0),
          ("early_postnatal", 20.0, 31.0),
          ("late_postnatal", 31.0, 76.0))


def stage_name(time: float) -> str:
    if time < 20.0:
        return "prenatal"
    if time < 31.0:
        return "early_postnatal"
    return "late_postnatal"


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(ranked)
    result[order] = np.minimum(ranked, 1.0)
    return result


def rank_stage_differences(expression: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Omnibus stage F test, conditional on additive strain intercepts."""
    times = metadata.time.to_numpy(dtype=float)
    stages = np.array([stage_name(time) for time in times])
    strain = metadata.strain.astype(str).to_numpy()
    baseline = np.column_stack((np.ones(len(metadata)), strain == "B6", strain == "C3H")).astype(float)
    full = np.column_stack((baseline, stages == "early_postnatal", stages == "late_postnatal")).astype(float)
    values = expression[metadata["sample"].tolist()].to_numpy(dtype=float).T
    if not np.isfinite(values).all():
        raise ValueError("Expression contains non-finite values; define an imputation policy before screening")
    residual_reduced = values - baseline @ np.linalg.lstsq(baseline, values, rcond=None)[0]
    residual_full = values - full @ np.linalg.lstsq(full, values, rcond=None)[0]
    rss_reduced = np.sum(residual_reduced ** 2, axis=0)
    rss_full = np.sum(residual_full ** 2, axis=0)
    numerator = np.maximum((rss_reduced - rss_full) / 2.0, 0.0)
    denominator = np.maximum(rss_full / (len(metadata) - full.shape[1]), 1e-15)
    statistic = numerator / denominator
    p_value = f.sf(statistic, 2, len(metadata) - full.shape[1])
    ranking = expression[["Gene ID", "Gene Symbol"]].copy()
    ranking["stage_F"] = statistic
    ranking["stage_p"] = p_value
    ranking["stage_q"] = bh_adjust(p_value)
    ranking["global_variance"] = np.var(values, axis=0, ddof=1)
    return ranking.sort_values(["stage_q", "stage_F", "Gene Symbol"],
                               ascending=[True, False, True], kind="stable").reset_index(drop=True)


def group_means(expression: pd.DataFrame, metadata: pd.DataFrame,
                genes: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    indexed = expression.set_index("Gene Symbol")
    matrix = indexed.loc[genes, metadata["sample"].tolist()].to_numpy(dtype=float)
    groups = []
    rows = []
    for (strain, stage, time), group in metadata.groupby(["strain", "stage_group", "time"], sort=True):
        indices = metadata.index.get_indexer(group.index)
        groups.append(matrix[:, indices].mean(axis=1))
        rows.append({"strain": strain, "stage": stage, "time": float(time),
                     "n_biological_samples": len(indices)})
    return np.stack(groups, axis=0), pd.DataFrame(rows)


def stage_correlations(values: np.ndarray, groups: pd.DataFrame,
                       genes: list[str], tf_genes: list[str], targets: list[str],
                       n_per_target: int) -> pd.DataFrame:
    index = {gene: i for i, gene in enumerate(genes)}
    tf_index = np.array([index[gene] for gene in tf_genes])
    target_index = np.array([index[gene] for gene in targets])
    rows = []
    for stage, _, _ in STAGES:
        keep = groups.stage.eq(stage).to_numpy()
        stage_values = values[keep].copy()
        stage_strain = groups.loc[keep, "strain"].to_numpy()
        for strain in np.unique(stage_strain):
            stage_values[stage_strain == strain] -= stage_values[stage_strain == strain].mean(axis=0)
        tf = stage_values[:, tf_index]
        target = stage_values[:, target_index]
        tf_norm = np.linalg.norm(tf, axis=0)
        target_norm = np.linalg.norm(target, axis=0)
        tf = tf / np.where(tf_norm > 1e-12, tf_norm, np.inf)
        target = target / np.where(target_norm > 1e-12, target_norm, np.inf)
        correlations = tf.T @ target
        correlations = np.clip(correlations, -1.0, 1.0)
        for j, gene in enumerate(targets):
            order = sorted(range(len(tf_genes)),
                           key=lambda i: (-abs(float(correlations[i, j])), tf_genes[i]))
            selected = [i for i in order if tf_genes[i] != gene][:n_per_target]
            for rank, i in enumerate(selected, start=1):
                rows.append({"stage": stage, "target": gene, "regulator": tf_genes[i],
                             "rank": rank, "within_strain_centered_correlation": float(correlations[i, j]),
                             "n_strain_time_means": int(keep.sum())})
    return pd.DataFrame(rows)


def all_tf_target_pairs(tf_genes: list[str], targets: list[str]) -> pd.DataFrame:
    """Allow every selected TF to affect every state gene except itself."""
    return pd.DataFrame(
        [(stage, target, regulator)
         for stage, _, _ in STAGES
         for target in targets
         for regulator in tf_genes
         if regulator != target],
        columns=["stage", "target", "regulator"],
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression", type=Path,
                        default=Path("data/raw/gse74243/GSE74243_matrix_RMAlog2expressionvalues.txt.gz"))
    parser.add_argument("--collectri", type=Path,
                        default=Path("data/external/collectri_mouse.tsv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/processed/three_stage_1000_candidates"))
    parser.add_argument("--n-targets", type=int, default=1000)
    parser.add_argument("--n-tfs", type=int, default=None,
                        help="Select this many measured CollecTRI TFs by the strain-adjusted stage F ranking; default keeps all")
    parser.add_argument("--max-tfs-per-target", type=int, default=20)
    parser.add_argument("--pair-mode", choices=("screened", "all_tf_targets"),
                        default="all_tf_targets",
                        help="Default: all selected TF-state-gene pairs; screened is for reproducing legacy results only")
    parser.add_argument("--include-regulator-dynamics", action="store_true",
                        help="Screen incoming TFs for all state genes, including regulators outside the top targets")
    parser.add_argument("--excluded-times", default="",
                        help="Comma-separated times to omit before all screening steps")
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    excluded = {float(value) for value in args.excluded_times.split(",") if value.strip()}
    expression, metadata = read_gse74243_expression(args.expression)
    metadata = metadata[metadata.time.gt(9.5) & ~metadata.time.isin(excluded)].copy().reset_index(drop=True)
    metadata["stage_group"] = metadata.time.map(stage_name)
    expression = expression[["Gene ID", "Gene Symbol", *metadata["sample"].tolist()]].copy()
    expression = expression.dropna(subset=["Gene Symbol"])
    expression["Gene Symbol"] = expression["Gene Symbol"].astype(str).str.strip()
    expression = expression[expression["Gene Symbol"].ne("")]
    expression["global_variance"] = expression[metadata["sample"].tolist()].var(axis=1, ddof=1)
    expression = expression.sort_values(["global_variance", "Gene Symbol", "Gene ID"],
                                        ascending=[False, True, True], kind="stable")
    expression = expression.drop_duplicates("Gene Symbol", keep="first").drop(columns="global_variance")
    ranking = rank_stage_differences(expression, metadata)
    if len(ranking) < args.n_targets:
        raise ValueError("Too few measured genes for requested target count")
    targets = ranking.head(args.n_targets)["Gene Symbol"].astype(str).tolist()
    catalogue = set(read_collectri(args.collectri).regulator.dropna().astype(str))
    measured_tf_genes = set(catalogue & set(expression["Gene Symbol"].astype(str)))
    ranked_tfs = ranking.loc[ranking["Gene Symbol"].isin(measured_tf_genes),
                             "Gene Symbol"].astype(str).tolist()
    if args.n_tfs is not None and (args.n_tfs < 1 or args.n_tfs > len(ranked_tfs)):
        raise ValueError(f"--n-tfs must be between 1 and {len(ranked_tfs)}")
    tf_genes = ranked_tfs[:args.n_tfs] if args.n_tfs is not None else ranked_tfs
    if args.pair_mode == "screened" and len(tf_genes) < args.max_tfs_per_target + 1:
        raise ValueError("Too few measured regulators for self-edge-free top-K selection")
    genes = sorted(set(targets) | set(tf_genes))
    means, groups = group_means(expression, metadata, genes)
    network_targets = genes if args.include_regulator_dynamics else targets
    candidate = (
        stage_correlations(means, groups, genes, tf_genes, network_targets,
                           args.max_tfs_per_target)
        if args.pair_mode == "screened"
        else all_tf_target_pairs(tf_genes, network_targets)
    )
    args.output_dir.mkdir(parents=True)
    ranking.to_csv(args.output_dir / "stage_differential_gene_ranking.tsv", sep="\t", index=False)
    pd.DataFrame({"target": targets, "rank": np.arange(1, len(targets) + 1)}).to_csv(
        args.output_dir / "selected_targets.tsv", sep="\t", index=False)
    ranking.loc[ranking["Gene Symbol"].isin(measured_tf_genes)].to_csv(
        args.output_dir / "measured_collectri_regulators.tsv", sep="\t", index=False)
    ranking.loc[ranking["Gene Symbol"].isin(tf_genes)].to_csv(
        args.output_dir / "selected_regulators.tsv", sep="\t", index=False)
    candidate.to_csv(args.output_dir / "candidate_tf_target_pairs.tsv.gz", sep="\t",
                     index=False, compression="gzip")
    groups.to_csv(args.output_dir / "strain_time_groups.tsv", sep="\t", index=False)
    np.savez_compressed(args.output_dir / "strain_time_expression.npz",
                        values=means, genes=np.asarray(genes, dtype=str),
                        strain=groups.strain.to_numpy(dtype=str),
                        stage=groups.stage.to_numpy(dtype=str),
                        time=groups.time.to_numpy(dtype=float))
    summary = {"n_training_samples": len(metadata), "excluded_times": sorted(excluded),
               "n_measured_genes": len(expression),
               "n_measured_collectri_regulators": len(measured_tf_genes),
               "n_selected_regulators": len(tf_genes),
               "n_selected_targets": len(targets),
               "n_global_state_genes": len(genes),
               "regulator_dynamics_included": args.include_regulator_dynamics,
               "n_network_target_equations": len(network_targets),
               "pair_mode": args.pair_mode,
               "max_candidate_tfs_per_target": args.max_tfs_per_target if args.pair_mode == "screened" else None,
               "n_candidate_pairs": len(candidate),
               "n_stage_significant_genes_q_lt_0_05": int((ranking.stage_q < 0.05).sum()),
               "stage_strain_time_groups": groups.groupby("stage").size().to_dict(),
               "interpretation": "Allowed TF-target pairs are parameter positions, not evidence of directed regulation. No GRN has been fitted.",
               "screening": ("Stage omnibus F test adjusted for strain; TF-target Pearson correlation of strain-centered strain-time means within each stage."
                             if args.pair_mode == "screened"
                             else "Stage omnibus F test adjusted for strain selects the gene panel and TFs; every selected TF-to-state-gene pair except self is allowed without correlation screening.")}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
