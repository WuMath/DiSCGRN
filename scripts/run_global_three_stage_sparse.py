#!/usr/bin/env python3
"""Fit three shared global DiSCGRN networks using sparse exponential actions.

AJ, B6, and C3H retain separate observed trajectories but share one A and b
within each prespecified stage. The model is the original coupled affine ODE,
not a collection of targetwise conditional equations. Stage boundaries are
fixed, so this run does not claim to detect change points.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from ctcpgrn.global_sparse_fit import GlobalTransitions, fit_global_sparse_stage
from ctcpgrn.io import read_gse74243_expression
from prepare_real_data_inputs import groups_and_splits, standardise_panel


STAGES = (("prenatal", 11.5, 20.0),
          ("early_postnatal", 20.0, 31.0),
          ("late_postnatal", 31.0, 76.0))


def load_stage_transitions(
    candidate_dir: Path, expression_path: Path,
    heldout_times: set[float] | None = None, partition: str = "train",
    interval_weight_power: float = 0.0,
) -> tuple[list[str], dict[str, GlobalTransitions]]:
    """Load stage transitions with fold-specific probe and scale selection."""
    if partition not in {"train", "test"}:
        raise ValueError("partition must be train or test")
    if not 0.0 <= interval_weight_power <= 1.0:
        raise ValueError("interval_weight_power must lie in [0, 1]")
    heldout_times = heldout_times or set()
    with np.load(candidate_dir / "strain_time_expression.npz", allow_pickle=False) as archive:
        genes = archive["genes"].astype(str).tolist()
    expression, metadata = read_gse74243_expression(expression_path)
    metadata = metadata[metadata.time.gt(9.5)].reset_index(drop=True)
    samples = metadata["sample"].tolist()
    ranking = pd.read_csv(candidate_dir / "stage_differential_gene_ranking.tsv", sep="\t")
    panel = ranking.set_index("Gene Symbol").loc[genes].reset_index()[["Gene ID", "Gene Symbol"]]
    selected = expression[["Gene ID", "Gene Symbol", *samples]].set_index("Gene ID").loc[
        panel["Gene ID"]].reset_index()
    training_samples = set(metadata.loc[~metadata.time.isin(heldout_times), "sample"])
    normalised, _ = standardise_panel(selected, metadata, panel, training_samples)
    groups, splits = groups_and_splits(normalised, metadata, heldout_times)
    stages = {}
    for name, left, right in STAGES:
        xs, ys, intervals, weights = [], [], [], []
        for group in groups:
            group_split = splits.loc[splits.strain.eq(group.name), "split"].to_numpy()
            keep = ((group.times[:-1] >= left) & (group.times[1:] <= right)
                    & (group_split == partition))
            xs.append(group.values[:-1][keep])
            ys.append(group.values[1:][keep])
            intervals.append(np.diff(group.times)[keep])
            interval = np.diff(group.times)[keep]
            weights.append(group.weights[1:][keep]
                           * interval[:, None] ** (-interval_weight_power))
        stages[name] = GlobalTransitions(np.concatenate(xs), np.concatenate(ys),
                                         np.concatenate(intervals), np.concatenate(weights))
    return genes, stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path,
                        default=Path("data/processed/three_stage_global_500_30tf_20261007"))
    parser.add_argument("--expression", type=Path,
                        default=Path("data/raw/gse74243/GSE74243_matrix_RMAlog2expressionvalues.txt.gz"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/global_three_stage_1000_20261007"))
    parser.add_argument("--stages", default="prenatal,early_postnatal,late_postnatal")
    parser.add_argument("--penalty", type=float, default=0.05)
    parser.add_argument("--ridge", type=float, default=0.0,
                        help="Nonnegative L2 coefficient on off-diagonal edges; zero reproduces the original L1 fit")
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--optimizer", choices=("proximal", "lbfgsb", "active_lbfgsb"),
                        default="proximal")
    parser.add_argument("--parameter-scale", type=float, default=1.0,
                        help="Positive numerical parameter scaling for L-BFGS-B; leaves the physical ODE and objective unchanged")
    parser.add_argument("--active-rounds", type=int, default=8)
    parser.add_argument("--active-add-per-round", type=int, default=200)
    parser.add_argument("--kkt-tolerance", type=float, default=1e-3)
    parser.add_argument("--objective-tolerance", type=float, default=1e-7)
    parser.add_argument("--resume-from", type=Path,
                        help="Directory with matching stage generators and baselines for continuation")
    parser.add_argument("--edge-threshold", type=float, default=1e-8)
    parser.add_argument("--interval-weight-power", type=float, default=0.0,
                        help="Sensitivity objective: multiply transition weights by delta_t raised to minus this power")
    parser.add_argument("--excluded-times", default="",
                        help="Diagnostic only: comma-separated observation times excluded from fitting and normalization")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    requested = args.stages.split(",")
    if any(name not in {item[0] for item in STAGES} for name in requested):
        raise ValueError("Unknown stage")
    summary = json.loads((args.candidate_dir / "summary.json").read_text())
    if not summary.get("regulator_dynamics_included"):
        raise ValueError("Global model requires a candidate equation for every state gene")
    torch.set_num_threads(1)
    excluded_times = {float(item) for item in args.excluded_times.split(",") if item.strip()}
    genes, stage_data = load_stage_transitions(
        args.candidate_dir, args.expression, heldout_times=excluded_times,
        interval_weight_power=args.interval_weight_power)
    index = {gene: i for i, gene in enumerate(genes)}
    candidate = pd.read_csv(args.candidate_dir / "candidate_tf_target_pairs.tsv.gz", sep="\t")
    args.output_dir.mkdir(parents=True)
    all_edges, diagnostics = [], []
    for name in requested:
        part = candidate[candidate.stage.eq(name)]
        rows = part.target.map(index).to_numpy(dtype=np.int64)
        cols = part.regulator.map(index).to_numpy(dtype=np.int64)
        table = stage_data[name]
        print(f"[{name}] {table.x.shape[1]} global states, {len(table.dt)} pooled transitions, "
              f"{len(rows)} candidate off-diagonal edges", flush=True)
        initial = None
        if args.resume_from is not None:
            old_a = sparse.load_npz(args.resume_from / f"{name}_generator.npz").tocsr()
            old_b = np.load(args.resume_from / f"{name}_baseline.npy", allow_pickle=False)
            if old_a.shape != (len(genes), len(genes)) or old_b.shape != (len(genes),):
                raise ValueError(f"Continuation dimensions differ for {name}")
            initial = (np.r_[np.asarray(old_a[rows, cols]).ravel(), old_a.diagonal()], old_b)
        def save_round_checkpoint(edge_values, intercept, entry):
            round_dir = (args.output_dir / "checkpoints" / name /
                         f"round_{entry['round']:02d}")
            round_dir.mkdir(parents=True, exist_ok=True)
            p_stage = len(genes)
            matrix = sparse.coo_matrix(
                (edge_values, (np.r_[rows, np.arange(p_stage)],
                               np.r_[cols, np.arange(p_stage)])),
                shape=(p_stage, p_stage)).tocsr()
            sparse.save_npz(round_dir / f"{name}_generator.npz", matrix)
            np.save(round_dir / f"{name}_baseline.npy", intercept)
            pd.DataFrame([entry]).to_csv(round_dir / "round_diagnostics.tsv",
                                         sep="\t", index=False)
            print(f"[{name}] active round {entry['round']} "
                  f"kkt_inf={entry['kkt_inf']:.4g} "
                  f"edges={entry['nonzero_offdiag']} "
                  f"objective={entry['objective']:.6g}", flush=True)
        started = time.perf_counter()
        fit = fit_global_sparse_stage(table, rows, cols, args.penalty,
                                      ridge=args.ridge,
                                      max_iter=args.max_iter,
                                      learning_rate=args.learning_rate,
                                      kkt_tolerance=args.kkt_tolerance,
                                      objective_tolerance=args.objective_tolerance,
                                      initial=initial, optimizer=args.optimizer,
                                      parameter_scale=args.parameter_scale,
                                      active_rounds=args.active_rounds,
                                      active_add_per_round=args.active_add_per_round,
                                      progress_callback=save_round_checkpoint
                                      if args.optimizer == "active_lbfgsb" else None)
        elapsed = time.perf_counter() - started
        p = len(genes)
        generator = sparse.coo_matrix((fit.edge_values,
                                       (np.r_[rows, np.arange(p)], np.r_[cols, np.arange(p)])),
                                      shape=(p, p)).tocsr()
        sparse.save_npz(args.output_dir / f"{name}_generator.npz", generator)
        np.save(args.output_dir / f"{name}_baseline.npy", fit.baseline)
        pd.DataFrame(fit.trace).to_csv(args.output_dir / f"{name}_optimization_trace.tsv",
                                       sep="\t", index=False)
        if fit.converged:
            for i, j, weight in zip(rows, cols, fit.edge_values[:len(rows)]):
                if abs(weight) > args.edge_threshold:
                    all_edges.append({"stage": name, "regulator": genes[j], "target": genes[i],
                                      "edge_weight": float(weight)})
        diagnostics.append({"stage": name, "n_state_genes": p,
                            "n_transitions": len(table.dt),
                            "n_candidate_edges": len(rows),
                            "n_selected_edges": fit.n_nonzero_offdiag,
                            "data_loss": fit.data_loss,
                            "ridge": args.ridge,
                            "objective": fit.objective,
                            "iterations": fit.iterations,
                            "converged": fit.converged,
                            "kkt_inf": fit.kkt_inf,
                            "runtime_seconds": elapsed})
        pd.DataFrame(diagnostics).to_csv(args.output_dir / "fit_diagnostics.tsv", sep="\t", index=False)
        pd.DataFrame(all_edges, columns=("stage", "regulator", "target", "edge_weight")).to_csv(
            args.output_dir / "selected_stage_edges.tsv.gz", sep="\t",
            index=False, compression="gzip")
        print(f"[{name}] selected={fit.n_nonzero_offdiag} objective={fit.objective:.6g} "
              f"iterations={fit.iterations} converged={fit.converged} "
              f"kkt_inf={fit.kkt_inf:.4g} "
              f"seconds={elapsed:.1f}", flush=True)
    (args.output_dir / "run_configuration.json").write_text(json.dumps({
        "candidate_dir": str(args.candidate_dir), "stages": requested,
        "penalty": args.penalty, "max_iter": args.max_iter,
        "ridge": args.ridge,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "parameter_scale": args.parameter_scale,
        "active_rounds": args.active_rounds,
        "active_add_per_round": args.active_add_per_round,
        "kkt_tolerance": args.kkt_tolerance,
        "objective_tolerance": args.objective_tolerance,
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "edge_threshold": args.edge_threshold,
        "interval_weight_power": args.interval_weight_power,
        "excluded_times": sorted(excluded_times),
        "model": "shared global affine ODE x'=A_k x+b_k, sparse exponential action",
        "caveat": "Fixed prespecified stages; penalty is exploratory until calibrated by independent blocked validation."
    }, indent=2) + "\n")
    if not all(item["converged"] for item in diagnostics):
        raise RuntimeError("At least one stage failed the KKT convergence criterion. "
                           "Saved matrices are continuation checkpoints only; "
                           "nonconverged edges were excluded from selected_stage_edges.tsv.gz.")


if __name__ == "__main__":
    main()
