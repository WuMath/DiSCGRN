#!/usr/bin/env python3
"""Fit shared change points and strain-specific networks to GSE74243."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ctcpgrn.external import (
    collectri_panel_edges,
    read_chipatlas_directory,
    read_collectri,
    validation_metrics,
)
from ctcpgrn.io import LongitudinalGroup
from ctcpgrn.segmentation import segment_shared_change_points


def load_groups(path: Path) -> tuple[list[str], list[LongitudinalGroup]]:
    archive = np.load(path, allow_pickle=False)
    genes = archive["genes"].astype(str).tolist()
    groups = []
    for name in ("AJ", "B6", "C3H"):
        groups.append(LongitudinalGroup(name, archive[f"{name}_times"], archive[f"{name}_values"], archive[f"{name}_weights"], archive[f"{name}_n_replicates"]))
    return genes, groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gse74243.yaml"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed/gse74243"))
    parser.add_argument("--collectri", type=Path, default=Path("data/external/collectri_mouse.tsv"))
    parser.add_argument("--chip-atlas", type=Path, default=Path("data/external/chipatlas_mm10"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/gse74243"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    genes, groups = load_groups(args.processed / "trajectories.npz")
    panel = pd.read_csv(args.processed / "gene_panel.tsv", sep="\t")
    regulators = panel["is_candidate_regulator"].astype(bool).to_numpy()
    p = len(genes)
    allowed = np.zeros((p, p), dtype=bool)
    allowed[:, regulators] = True
    np.fill_diagonal(allowed, True)
    union_times = np.unique(np.concatenate([group.times for group in groups]))
    lambda_grid = config["lambda_grid"][:1] if args.quick else config["lambda_grid"]
    beta_grid = config["beta_grid"][:1] if args.quick else config["beta_grid"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_all = read_collectri(args.collectri)
    evidence_panel = collectri_panel_edges(evidence_all, genes)
    chip = read_chipatlas_directory(args.chip_atlas)
    chip_panel = chip[chip["regulator"].isin(genes) & chip["target"].isin(genes)].copy()
    grid_records = []
    best = None
    for lambda_l1 in lambda_grid:
        for beta in beta_grid:
            result = segment_shared_change_points(
                groups, union_times, lambda_l1, beta,
                minimum_transitions=config["minimum_transitions"],
                max_changes=1 if args.quick else config["max_changes"],
                allowed_edges=allowed,
                fitter_kwargs={"max_iter": 20 if args.quick else 500},
            )
            record = {
                "lambda": lambda_l1, "beta": beta, "objective": result.objective,
                "change_point_times": result.change_point_times,
                "segments": result.segments,
            }
            grid_records.append(record)
            if best is None or result.objective < best[0]:
                best = (result.objective, lambda_l1, beta, result)
    assert best is not None
    _, best_lambda, best_beta, result = best
    matrices = {}
    external = {}
    for segment_index, fit_map in enumerate(result.fits):
        for strain, fit in fit_map.items():
            key = f"segment_{segment_index}_{strain}"
            matrices[key] = fit.matrix
            external[key] = validation_metrics(fit.matrix, genes, evidence_panel)
    np.savez_compressed(args.output_dir / "networks.npz", genes=np.asarray(genes), **matrices)
    summary = {
        "selected_lambda": best_lambda,
        "selected_beta": best_beta,
        "change_point_times": result.change_point_times,
        "segments": result.segments,
        "collectri_panel_edges": len(evidence_panel),
        "chip_atlas_panel_pairs_downloaded": len(chip_panel),
        "chip_atlas_lung_supported_pairs": int(chip_panel["lung_support"].sum()) if len(chip_panel) else 0,
        "external_validation": external,
        "grid": grid_records,
        "selection_warning": "Minimum in-sample objective is a pipeline default; final paper must use blocked CV and one-standard-error selection.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("selected_lambda", "selected_beta", "change_point_times", "collectri_panel_edges")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
