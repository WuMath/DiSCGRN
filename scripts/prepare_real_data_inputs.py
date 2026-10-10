#!/usr/bin/env python3
"""Prepare leakage-aware real-data inputs and independent evaluation evidence.

Run from the repository root after downloading the real-data profiles. This
script does not fit a network or interpret unreported edges as negative truth.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

from ctcpgrn.external import read_chipatlas_directory, read_collectri
from ctcpgrn.io import LongitudinalGroup, read_gse74243_expression


STRAINS = ("AJ", "B6", "C3H")
# This lung-development regulator is fixed before fitting because an
# independent P5 AT1 knockout and lung ChIP resource are available for it.
ANCHOR_REGULATORS = ("Nkx2-1",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_panel(
    expression: pd.DataFrame,
    training_samples: list[str],
    regulator_catalogue: set[str],
    n_genes: int,
    n_regulators: int,
    anchor_regulators: tuple[str, ...] = ANCHOR_REGULATORS,
) -> pd.DataFrame:
    """Select symbols and duplicate probes using training samples only."""
    if not 0 < n_regulators <= n_genes:
        raise ValueError("Require 0 < n_regulators <= n_genes")
    work = expression[["Gene ID", "Gene Symbol", *training_samples]].copy()
    work = work[work["Gene Symbol"].notna()].copy()
    work["Gene Symbol"] = work["Gene Symbol"].astype(str).str.strip()
    work = work[work["Gene Symbol"].ne("")].copy()
    values = work[training_samples].to_numpy(dtype=float)
    work["selection_variance"] = np.nanvar(values, axis=1, ddof=1)
    work = work[np.isfinite(work["selection_variance"])].copy()
    work = work.sort_values(["selection_variance", "Gene Symbol", "Gene ID"],
                            ascending=[False, True, True], kind="stable")
    work = work.drop_duplicates("Gene Symbol", keep="first")
    anchored = work[work["Gene Symbol"].isin(anchor_regulators)]
    if len(anchored) != len(anchor_regulators):
        raise ValueError(f"Missing pre-specified regulator(s): {anchor_regulators}")
    other_regulators = work[
        work["Gene Symbol"].isin(regulator_catalogue)
        & ~work["Gene Symbol"].isin(anchor_regulators)
    ].head(n_regulators - len(anchored))
    regulators = pd.concat([anchored, other_regulators])
    if len(regulators) != n_regulators:
        raise ValueError(f"Only {len(regulators)} measured candidate regulators")
    targets = work[~work["Gene Symbol"].isin(regulators["Gene Symbol"])].head(n_genes - n_regulators)
    if len(targets) != n_genes - n_regulators:
        raise ValueError("Not enough measured genes for the requested panel")
    panel = pd.concat([regulators, targets], ignore_index=True)
    panel["is_candidate_regulator"] = panel["Gene Symbol"].isin(set(regulators["Gene Symbol"]))
    panel["pre_specified_anchor"] = panel["Gene Symbol"].isin(anchor_regulators)
    return panel[["Gene ID", "Gene Symbol", "is_candidate_regulator", "pre_specified_anchor", "selection_variance"]]


def standardise_panel(
    expression: pd.DataFrame, metadata: pd.DataFrame, panel: pd.DataFrame,
    training_samples: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute strain-specific means and scales without held-out samples."""
    columns = metadata["sample"].tolist()
    selected = expression.set_index("Gene Symbol").loc[panel["Gene Symbol"], columns].copy()
    selected.index = panel["Gene Symbol"].tolist()
    records = []
    for strain in STRAINS:
        group_meta = metadata[metadata["strain"].eq(strain)]
        all_cols = group_meta["sample"].tolist()
        train_cols = [column for column in all_cols if column in training_samples]
        if len(train_cols) < 2:
            raise ValueError(f"Too few training samples for {strain}")
        fit_values = selected[train_cols].to_numpy(dtype=float)
        means = np.nanmean(fit_values, axis=1)
        scales = np.nanstd(fit_values, axis=1, ddof=1)
        scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
        selected.loc[:, all_cols] = (selected[all_cols].to_numpy(dtype=float) - means[:, None]) / scales[:, None]
        records.extend(
            {"strain": strain, "gene": gene, "train_mean_log2_rma": float(mean),
             "train_sd_log2_rma": float(scale), "n_train_samples": len(train_cols)}
            for gene, mean, scale in zip(selected.index, means, scales)
        )
    if not np.isfinite(selected.to_numpy(dtype=float)).all():
        raise ValueError("Selected expression panel contains non-finite values")
    return selected, pd.DataFrame(records)


def groups_and_splits(
    normalised: pd.DataFrame, metadata: pd.DataFrame,
    heldout_times: set[float],
) -> tuple[list[LongitudinalGroup], pd.DataFrame]:
    """Create trajectories and transition masks on each strain's own time grid.

    A transition is held out if its endpoint is in the test block. The exit
    transition, from the final held-out point to a training point, is embargoed.
    This prevents a fit from using a held-out state as either endpoint.
    """
    groups = []
    split_rows = []
    for strain in STRAINS:
        strain_meta = metadata[metadata["strain"].eq(strain)]
        times, means, variances, ns = [], [], [], []
        for time, subset in strain_meta.groupby("time", sort=True):
            values = normalised[subset["sample"].tolist()].to_numpy(dtype=float)
            times.append(float(time))
            means.append(values.mean(axis=1))
            variances.append(values.var(axis=1, ddof=1) if values.shape[1] > 1 else np.full(values.shape[0], np.nan))
            ns.append(values.shape[1])
        times_arr = np.asarray(times, dtype=float)
        mean_arr = np.stack(means)
        variance_arr = np.stack(variances)
        n_arr = np.asarray(ns, dtype=float)
        train_time = ~np.isin(times_arr, list(heldout_times))
        available = variance_arr[train_time]
        available = available[np.isfinite(available) & (available > 0)]
        floor = float(np.quantile(available, 0.25)) if available.size else 1.0
        variance_arr = np.where(np.isfinite(variance_arr), np.maximum(variance_arr, floor), floor)
        weights = n_arr[:, None] / variance_arr
        weights /= np.mean(weights[train_time])
        groups.append(LongitudinalGroup(strain, times_arr, mean_arr, weights, n_arr))
        for i in range(len(times_arr) - 1):
            start_held = bool(times_arr[i] in heldout_times)
            end_held = bool(times_arr[i + 1] in heldout_times)
            split = "test" if end_held else ("embargo" if start_held else "train")
            split_rows.append({"strain": strain, "transition_index": i,
                               "start_time": times_arr[i], "end_time": times_arr[i + 1],
                               "split": split})
    return groups, pd.DataFrame(split_rows)


def write_model_bundle(
    destination: Path, expression: pd.DataFrame, metadata: pd.DataFrame,
    catalogue: set[str], n_genes: int, n_regulators: int,
    heldout_times: set[float],
    anchor_regulators: tuple[str, ...] = ANCHOR_REGULATORS,
) -> pd.DataFrame:
    destination.mkdir(parents=True, exist_ok=True)
    training_meta = metadata[~metadata["time"].isin(heldout_times)]
    train_samples = training_meta["sample"].tolist()
    panel = choose_panel(expression, train_samples, catalogue, n_genes,
                         n_regulators, anchor_regulators)
    unique_expression = expression.loc[expression["Gene ID"].isin(panel["Gene ID"])].copy()
    # The chosen Gene ID, not a symbol-only lookup, fixes the representative
    # probe when the original matrix contains duplicate symbols.
    unique_expression = unique_expression.drop_duplicates("Gene ID").set_index("Gene ID")
    unique_expression = unique_expression.loc[panel["Gene ID"]].reset_index()
    normalised, scales = standardise_panel(unique_expression, metadata, panel, set(train_samples))
    groups, splits = groups_and_splits(normalised, metadata, heldout_times)
    panel.to_csv(destination / "gene_panel.tsv", sep="\t", index=False)
    metadata.to_csv(destination / "sample_metadata.tsv", sep="\t", index=False)
    scales.to_csv(destination / "normalisation.tsv", sep="\t", index=False)
    splits.to_csv(destination / "transition_split.tsv", sep="\t", index=False)
    normalised.rename_axis("gene").reset_index().to_csv(
        destination / "sample_expression_zscore.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    # NumPy object arrays would require allow_pickle=True in the existing
    # model loader. Persist a Unicode array instead.
    payload: dict[str, np.ndarray] = {"genes": np.asarray(panel["Gene Symbol"].astype(str).tolist(), dtype=str)}
    for group in groups:
        payload[f"{group.name}_times"] = group.times
        payload[f"{group.name}_values"] = group.values
        payload[f"{group.name}_weights"] = group.weights
        payload[f"{group.name}_n_replicates"] = group.n_replicates
        group_splits = splits[splits["strain"].eq(group.name)]["split"].to_numpy()
        payload[f"{group.name}_train_transition_mask"] = group_splits == "train"
        payload[f"{group.name}_test_transition_mask"] = group_splits == "test"
    np.savez_compressed(destination / "trajectories.npz", **payload)
    (destination / "split_info.json").write_text(json.dumps({
        "heldout_times": sorted(heldout_times),
        "n_train_samples": len(train_samples),
        "n_test_transitions": int(splits["split"].eq("test").sum()),
        "n_embargo_transitions": int(splits["split"].eq("embargo").sum()),
        "conditional_one_step_prediction": True,
        "warning": "Existing run_gse74243.py ignores transition masks; use a mask-aware fitting runner for blocked CV.",
    }, indent=2) + "\n", encoding="utf-8")
    return panel


def make_cv_blocks(times: np.ndarray, n_folds: int) -> list[set[float]]:
    if n_folds < 2:
        raise ValueError("Need at least two blocked folds")
    # Leave the first and last two observed stages outside every test block.
    candidate = times[2:-2]
    if len(candidate) < 2 * n_folds:
        raise ValueError("Not enough time points for the requested folds")
    return [set(map(float, block)) for block in np.array_split(candidate, n_folds)]


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("BH adjustment requires finite p values in [0, 1]")
    order = np.argsort(values, kind="stable")
    ranks = np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate((values[order] * len(values) / ranks)[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def prepare_knockout(path: Path) -> pd.DataFrame:
    """Use only the six matched P5 AT1 control/mutant columns from GSE129583."""
    table = pd.read_excel(path, sheet_name="Sheet1")
    controls = [f"Nkx2-1;Aqp5 P5 AT1 Control_{letter}" for letter in "ABC"]
    mutants = [f"Nkx2-1;Aqp5 P5 AT1 Mutant_{letter}" for letter in "ABC"]
    required = ["gene_short_name", *controls, *mutants]
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f"GSE129583 workbook is missing {sorted(missing)}")
    work = table[required].copy()
    work = work[work["gene_short_name"].map(lambda value: isinstance(value, str) and bool(value.strip()))].copy()
    work["gene"] = work["gene_short_name"].str.strip()
    numeric = work[controls + mutants].apply(pd.to_numeric, errors="coerce")
    work = work[numeric.notna().all(axis=1) & numeric.ge(0).all(axis=1)].copy()
    work[controls + mutants] = numeric.loc[work.index]
    work["mean_fpkm"] = work[controls + mutants].mean(axis=1)
    work = work.sort_values(["mean_fpkm", "gene"], ascending=[False, True], kind="stable")
    work = work.drop_duplicates("gene", keep="first").sort_values("gene")
    control_log = np.log2(1.0 + work[controls].to_numpy(dtype=float))
    mutant_log = np.log2(1.0 + work[mutants].to_numpy(dtype=float))
    effects = mutant_log.mean(axis=1) - control_log.mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p_values = stats.ttest_ind(mutant_log, control_log, axis=1, equal_var=False).pvalue
    p_values = np.where(np.isfinite(p_values), p_values, 1.0)
    result = pd.DataFrame({
        "gene": work["gene"].to_numpy(), "n_control": len(controls),
        "n_mutant": len(mutants), "mean_control_log2_fpkm1": control_log.mean(axis=1),
        "mean_mutant_log2_fpkm1": mutant_log.mean(axis=1),
        "mutant_minus_control_log2_fpkm1": effects,
        "abs_response": np.abs(effects), "welch_p": p_values,
        "mean_fpkm": work["mean_fpkm"].to_numpy(),
    })
    result["bh_q"] = bh_adjust(result["welch_p"].to_numpy())
    return result.sort_values("gene").reset_index(drop=True)


def prepare_evidence(
    panel: pd.DataFrame, collectri: pd.DataFrame, chip: pd.DataFrame,
) -> pd.DataFrame:
    """Annotate all assayed candidate pairs; unreported is not a negative label."""
    genes = panel["Gene Symbol"].astype(str).tolist()
    regulators = panel.loc[panel["is_candidate_regulator"], "Gene Symbol"].astype(str).tolist()
    reported = collectri[collectri["regulator"].isin(regulators) & collectri["target"].isin(genes)]
    signs = reported.groupby(["regulator", "target"])["sign"].agg(
        lambda values: int(next(iter(set(values)))) if len(set(values)) == 1 else 0
    )
    assayed = set(chip["regulator"])
    lung_assayed = set(chip.loc[chip["has_lung_experiment"], "regulator"])
    chip = chip[chip["regulator"].isin(regulators) & chip["target"].isin(genes)]
    chip = chip.sort_values("average_score", ascending=False).drop_duplicates(["regulator", "target"])
    chip_lookup = chip.set_index(["regulator", "target"])
    gene_index = {gene: index for index, gene in enumerate(genes)}
    records = []
    for regulator in regulators:
        for target in genes:
            if regulator == target:
                continue
            pair = (regulator, target)
            chip_row = chip_lookup.loc[pair] if pair in chip_lookup.index else None
            records.append({
                "regulator": regulator, "target": target,
                "regulator_index": gene_index[regulator], "target_index": gene_index[target],
                "collectri_reported": pair in signs.index,
                "collectri_sign": int(signs.loc[pair]) if pair in signs.index else pd.NA,
                "chip_regulator_assayed": regulator in assayed,
                "chip_cross_tissue_support": bool(chip_row["cross_tissue_support"]) if chip_row is not None else False,
                "chip_lung_experiment": regulator in lung_assayed,
                "chip_lung_support": bool(chip_row["lung_support"]) if chip_row is not None else False,
                "chip_lung_score": float(chip_row["lung_score"]) if chip_row is not None else np.nan,
            })
    return pd.DataFrame(records)


def _decode(values: np.ndarray) -> list[str]:
    return [item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item) for item in values]


def prepare_single_cell(archive_path: Path, destination: Path) -> dict:
    """Generate stage-level pseudobulk, not unobserved cell-type proportions."""
    import h5py

    destination.mkdir(parents=True, exist_ok=True)
    expression: dict[str, pd.Series] = {}
    qc = []
    with tarfile.open(archive_path, "r:") as archive:
        members = [m for m in archive.getmembers() if m.name.endswith("filtered_gene_bc_matrices.h5")]
        for member in members:
            stage_match = re.search(r"_(E\d+(?:\.\d+)?|P\d+)_", member.name)
            if stage_match is None:
                continue  # Adult 2-month samples are not assigned an arbitrary day.
            stage = stage_match.group(1)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot read {member.name}")
            with h5py.File(io.BytesIO(source.read()), "r") as handle:
                genome = handle["mm10"]
                matrix = sparse.csc_matrix(
                    (genome["data"][:], genome["indices"][:], genome["indptr"][:]),
                    shape=tuple(genome["shape"][:]),
                )
                names = _decode(genome["gene_names"][:])
            library = np.asarray(matrix.sum(axis=0)).ravel()
            detected = np.diff(matrix.indptr)
            mito = np.asarray(matrix[np.asarray([name.lower().startswith("mt-") for name in names])].sum(axis=0)).ravel()
            mito_fraction = np.divide(mito, library, out=np.ones_like(mito, dtype=float), where=library > 0)
            keep = (library >= 500) & (detected >= 200) & (mito_fraction <= 0.20)
            if not np.any(keep):
                raise ValueError(f"No single cells passed QC for {stage}")
            summed = np.asarray(matrix[:, keep].sum(axis=1)).ravel().astype(float)
            counts = pd.Series(summed, index=names).groupby(level=0).sum()
            cpm = counts / counts.sum() * 1_000_000.0
            expression[stage] = np.log2(cpm + 1.0)
            qc.append({"stage": stage, "n_cells_raw": matrix.shape[1],
                       "n_cells_qc": int(keep.sum()), "n_genes": matrix.shape[0],
                       "median_umis_qc": float(np.median(library[keep])),
                       "median_mito_fraction_qc": float(np.median(mito_fraction[keep]))})
    if not expression:
        raise ValueError("No developmental scRNA-seq matrices found in the archive")
    pd.DataFrame(expression).rename_axis("gene").reset_index().to_csv(
        destination / "stage_pseudobulk_log2cpm.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    pd.DataFrame(qc).to_csv(destination / "cell_qc_by_stage.tsv", sep="\t", index=False)
    return {"stages": sorted(expression), "n_gene_union": len(set().union(*(set(series.index) for series in expression.values())))}


def make_manifest(output: Path, inputs: list[Path]) -> None:
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json")
    payload = {
        "inputs": [{"path": str(path), "bytes": path.stat().st_size} for path in inputs],
        "outputs": [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                     "sha256": sha256(path)} for path in files],
        "interpretation": "Unreported regulatory pairs are unknown, not confirmed negative edges.",
    }
    (output / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/real_study"))
    parser.add_argument("--n-genes", type=int, default=50)
    parser.add_argument("--n-regulators", type=int, default=20)
    parser.add_argument("--no-anchor-regulators", action="store_true",
                        help="Select every regulator by training variance without forcing NKX2-1")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--with-single-cell", action="store_true")
    args = parser.parse_args()
    root, output = args.data_root, args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Choose a new --output-dir "
            "to keep each preparation run auditable."
        )
    expression_path = root / "raw/gse74243/GSE74243_matrix_RMAlog2expressionvalues.txt.gz"
    collectri_path = root / "external/collectri_mouse.tsv"
    chip_dir = root / "external/chipatlas_mm10"
    knockout_path = root / "raw/gse129583/GSE129583_processeddata_matrix_Nkx2-1_RNAseq.xlsx"
    inputs = [expression_path, collectri_path, knockout_path, *sorted(chip_dir.glob("*.tsv"))]
    if not chip_dir.is_dir() or not list(chip_dir.glob("*.tsv")):
        raise FileNotFoundError(f"No ChIP-Atlas target tables found in {chip_dir}")
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    expression, metadata = read_gse74243_expression(expression_path)
    metadata = metadata[metadata["time"] > 9.5].copy().reset_index(drop=True)
    expression = expression[["Gene ID", "Gene Symbol", *metadata["sample"].tolist()]].copy()
    collectri = read_collectri(collectri_path)
    catalogue = set(collectri["regulator"].dropna().astype(str))
    anchors = () if args.no_anchor_regulators else ANCHOR_REGULATORS
    chip = read_chipatlas_directory(chip_dir)
    full_panel = write_model_bundle(output / "model/full", expression, metadata, catalogue,
                                    args.n_genes, args.n_regulators, set(), anchors)
    union_times = np.unique(metadata["time"].to_numpy(dtype=float))
    folds = make_cv_blocks(union_times, args.cv_folds)
    fold_descriptions = []
    for number, heldout in enumerate(folds, start=1):
        fold_name = f"fold_{number:02d}"
        fold_panel = write_model_bundle(output / f"model/cv/{fold_name}", expression, metadata,
                                        catalogue, args.n_genes, args.n_regulators, heldout, anchors)
        evidence = prepare_evidence(fold_panel, collectri, chip)
        evidence.to_csv(output / f"model/cv/{fold_name}/edge_evidence.tsv", sep="\t", index=False)
        fold_descriptions.append({"fold": fold_name, "heldout_times": sorted(heldout),
                                  "n_panel_genes": len(fold_panel)})
    eval_dir = output / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    edge_evidence = prepare_evidence(full_panel, collectri, chip)
    edge_evidence.to_csv(eval_dir / "edge_evidence.tsv", sep="\t", index=False)
    knockout = prepare_knockout(knockout_path)
    knockout.to_csv(eval_dir / "nkx21_p5_at1_knockout_response.tsv", sep="\t", index=False)
    knockout[knockout["gene"].isin(full_panel["Gene Symbol"])].to_csv(
        eval_dir / "nkx21_panel_knockout_response.tsv", sep="\t", index=False
    )
    sc_summary = None
    if args.with_single_cell:
        sc_path = root / "raw/gse149563/GSE149563_RAW.tar"
        if not sc_path.is_file():
            raise FileNotFoundError(sc_path)
        sc_summary = prepare_single_cell(sc_path, output / "evaluation/single_cell")
        inputs.append(sc_path)
    summary = {
        "model_genes": len(full_panel), "candidate_regulators": int(full_panel["is_candidate_regulator"].sum()),
        "anchor_regulators": list(anchors),
        "gse74243_samples_after_excluding_whole_embryo_e9_5": len(metadata),
        "gse74243_times_by_strain": metadata.groupby("strain")["time"].nunique().to_dict(),
        "edge_universe_pairs": len(edge_evidence),
        "collectri_reported_pairs": int(edge_evidence["collectri_reported"].sum()),
        "chip_lung_supported_pairs": int(edge_evidence["chip_lung_support"].sum()),
        "knockout_tested_genes": len(knockout),
        "knockout_genes_in_model_panel": int(knockout["gene"].isin(full_panel["Gene Symbol"]).sum()),
        "cv_folds": fold_descriptions,
        "single_cell": sc_summary,
        "evaluation_warning": "Evidence coverage is incomplete; no full-network AUROC/AUPR or change-point sensitivity can be computed from these real datasets.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    make_manifest(output, inputs)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
