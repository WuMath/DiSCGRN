#!/usr/bin/env python3
"""Training-only alpha/ridge screening for the fixed three-stage global ODE.

This is a development-stage screen, not a final performance estimate.
Each fold reselects genes, regulators and candidate parents without its
held-out time points. The fit uses only transitions with both endpoints in
the training portion. Conditional one-step RMSE is evaluated at held-out
endpoints from their observed preceding state.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ctcpgrn.global_sparse_fit import fit_global_sparse_stage
from ctcpgrn.sparse_expm_action import sparse_affine_expm_action
from prepare_three_stage_1000_candidates import main as prepare_candidates
from run_global_three_stage_sparse import STAGES, load_stage_transitions


def predictions(table, rows, cols, edge_values, baseline):
    p = table.x.shape[1]
    edge_rows = torch.as_tensor(np.r_[rows, np.arange(p)], dtype=torch.long)
    edge_cols = torch.as_tensor(np.r_[cols, np.arange(p)], dtype=torch.long)
    values = torch.as_tensor(edge_values, dtype=torch.float64)
    intercept = torch.as_tensor(baseline, dtype=torch.float64)
    out = np.empty_like(table.y)
    with torch.no_grad():
        for interval in np.unique(table.dt):
            keep = table.dt == interval
            x = torch.as_tensor(table.x[keep], dtype=torch.float64)
            out[keep] = sparse_affine_expm_action(
                x, edge_rows, edge_cols, values, intercept, float(interval)).numpy()
    return out


def parse_folds(value: str) -> list[set[float]]:
    folds = [{float(item) for item in fold.split(",") if item.strip()}
             for fold in value.split(";")]
    if not folds or any(not fold for fold in folds):
        raise ValueError("Each fold needs at least one held-out time")
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression", type=Path,
                        default=Path("data/raw/gse74243/GSE74243_matrix_RMAlog2expressionvalues.txt.gz"))
    parser.add_argument("--collectri", type=Path,
                        default=Path("data/external/collectri_mouse.tsv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/global_three_stage_500_30tf_alpha_calibration"))
    parser.add_argument("--alphas", default="0.08,0.15,0.30")
    parser.add_argument("--ridges", default="0.0",
                        help="Comma-separated L2 coefficients, jointly screened with alpha")
    parser.add_argument("--stages", default="prenatal,early_postnatal,late_postnatal",
                        help="Comma-separated fixed stages to screen")
    parser.add_argument("--fold-times", default="16.5,24,38;18.5,27,44")
    parser.add_argument("--precomputed-fold-dirs", default="",
                        help="Comma-separated training-only candidate directories, one per fold")
    parser.add_argument("--n-targets", type=int, default=500)
    parser.add_argument("--n-tfs", type=int, default=30)
    parser.add_argument("--max-tfs-per-target", type=int, default=20)
    parser.add_argument("--screen-iter", type=int, default=80)
    parser.add_argument("--optimizer", choices=("proximal", "lbfgsb", "active_lbfgsb"),
                        default="lbfgsb")
    parser.add_argument("--active-rounds", type=int, default=12)
    parser.add_argument("--active-add-per-round", type=int, default=100)
    parser.add_argument("--interval-weight-power", type=float, default=0.0)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    alphas = [float(item) for item in args.alphas.split(",")]
    if not alphas or any(value <= 0 for value in alphas):
        raise ValueError("Positive alpha grid required")
    ridges = [float(item) for item in args.ridges.split(",")]
    if not ridges or any(value < 0 or not np.isfinite(value) for value in ridges):
        raise ValueError("Finite nonnegative ridge grid required")
    folds = parse_folds(args.fold_times)
    selected_stages = [item.strip() for item in args.stages.split(",")]
    if not selected_stages or len(set(selected_stages)) != len(selected_stages) or any(
        stage not in {item[0] for item in STAGES} for stage in selected_stages
    ):
        raise ValueError("--stages must contain distinct known stage names")
    precomputed = ([Path(item) for item in args.precomputed_fold_dirs.split(",") if item.strip()]
                   if args.precomputed_fold_dirs else [])
    if precomputed and len(precomputed) != len(folds):
        raise ValueError("Need one precomputed candidate directory per fold")
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    records = []
    scales = []
    for fold_index, heldout in enumerate(folds, start=1):
        fold_name = f"fold_{fold_index:02d}"
        candidate_dir = (precomputed[fold_index - 1] if precomputed else
                         args.output_dir / "fold_inputs" / fold_name)
        if not precomputed:
            prepare_candidates([
                "--expression", str(args.expression), "--collectri", str(args.collectri),
                "--output-dir", str(candidate_dir), "--n-targets", str(args.n_targets),
                "--n-tfs", str(args.n_tfs), "--max-tfs-per-target", str(args.max_tfs_per_target),
                "--include-regulator-dynamics",
                "--excluded-times", ",".join(map(str, sorted(heldout))),
            ])
        else:
            candidate_summary = json.loads((candidate_dir / "summary.json").read_text())
            if candidate_summary["excluded_times"] != sorted(heldout):
                raise ValueError(f"{candidate_dir}: excluded times do not match {fold_name}")
            if (candidate_summary["n_selected_targets"] != args.n_targets or
                    candidate_summary["n_selected_regulators"] != args.n_tfs or
                    candidate_summary.get("pair_mode") != "all_tf_targets"):
                raise ValueError(f"{candidate_dir}: candidate settings do not match")
        genes, train = load_stage_transitions(candidate_dir, args.expression,
                                              heldout, "train",
                                              args.interval_weight_power)
        test_genes, test = load_stage_transitions(candidate_dir, args.expression,
                                                 heldout, "test",
                                                 args.interval_weight_power)
        if genes != test_genes:
            raise AssertionError("Train and test gene order differs")
        index = {gene: i for i, gene in enumerate(genes)}
        candidates = pd.read_csv(candidate_dir / "candidate_tf_target_pairs.tsv.gz", sep="\t")
        for stage in selected_stages:
            training = train[stage]
            held_out = test[stage]
            if len(training.dt) < 3 or len(held_out.dt) < 1:
                raise ValueError(f"{fold_name}/{stage}: insufficient train or test transitions")
            part = candidates[candidates.stage.eq(stage)]
            rows = part.target.map(index).to_numpy(dtype=np.int64)
            cols = part.regulator.map(index).to_numpy(dtype=np.int64)
            scales.append({"fold": fold_name, "stage": stage,
                           "heldout_times": ",".join(map(str, sorted(heldout))),
                           "n_state_genes": len(genes),
                           "n_train_transitions": len(training.dt),
                           "n_test_transitions": len(held_out.dt),
                           "n_candidate_edges": len(rows),
                           "dt_min": float(training.dt.min()),
                           "dt_median": float(np.median(training.dt)),
                           "dt_max": float(training.dt.max()),
                           "x_sd": float(training.x.std()),
                           "x_abs_max": float(np.abs(training.x).max()),
                           "weight_median": float(np.median(training.weights)),
                           "weight_max": float(training.weights.max())})
            for alpha in alphas:
                for ridge in ridges:
                    started = time.perf_counter()
                    fit = fit_global_sparse_stage(
                        training, rows, cols, alpha, ridge=ridge,
                        max_iter=args.screen_iter, optimizer=args.optimizer,
                        active_rounds=args.active_rounds,
                        active_add_per_round=args.active_add_per_round)
                    predicted = predictions(held_out, rows, cols, fit.edge_values,
                                            fit.baseline)
                    rmse = float(np.sqrt(np.mean((predicted - held_out.y) ** 2)))
                    row = {"fold": fold_name, "stage": stage, "alpha": alpha,
                           "ridge": ridge, "test_rmse": rmse,
                           "n_selected_edges": fit.n_nonzero_offdiag,
                           "kkt_inf": fit.kkt_inf,
                           "converged": fit.converged,
                           "optimizer_iterations": fit.iterations,
                           "train_objective": fit.objective,
                           "runtime_seconds": time.perf_counter() - started}
                    records.append(row)
                    pd.DataFrame(records).to_csv(args.output_dir / "alpha_grid.tsv",
                                                  sep="\t", index=False)
                    print(f"[{fold_name}/{stage}] alpha={alpha:g} ridge={ridge:g} "
                          f"rmse={rmse:.4f} edges={fit.n_nonzero_offdiag} "
                          f"kkt={fit.kkt_inf:.4g}", flush=True)
    pd.DataFrame(scales).to_csv(args.output_dir / "scale_diagnostics.tsv",
                                sep="\t", index=False)
    grid = pd.DataFrame(records)
    summary = grid.groupby(["alpha", "ridge"]).agg(
        mean_test_rmse=("test_rmse", "mean"),
        max_kkt_inf=("kkt_inf", "max"),
        min_selected_edges=("n_selected_edges", "min"),
        n_converged=("converged", "sum"),
        n_fits=("converged", "size"),
    ).reset_index()
    summary["nonempty_all_stages_and_folds"] = summary.min_selected_edges.gt(0)
    summary.to_csv(args.output_dir / "alpha_summary.tsv", sep="\t", index=False)
    eligible = summary[
        summary.nonempty_all_stages_and_folds & summary.n_converged.eq(summary.n_fits)
    ]
    chosen = None if eligible.empty else eligible.sort_values(
        ["mean_test_rmse", "alpha", "ridge"], kind="stable").iloc[0]
    selection = {"alpha": None if chosen is None else float(chosen.alpha),
                 "ridge": None if chosen is None else float(chosen.ridge),
                 "status": "selected_from_converged_fits" if chosen is not None else "no_converged_setting",
                 "rule": "Lowest mean held-out conditional RMSE among alpha-ridge pairs with nonempty fitted networks in every fold and stage",
                 "warning": "Only settings converged in every fold and stage are eligible. These folds are development data, not an independent test.",
                 "folds": [sorted(fold) for fold in folds],
                 "stages": selected_stages,
                 "interval_weight_power": args.interval_weight_power,
                 "screen_iter": args.screen_iter}
    (args.output_dir / "selected_alpha.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
