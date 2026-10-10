#!/usr/bin/env python3
"""Compute a fixed-P0 rho profile with one shared candidate universe.

This reuses the original coupled affine ODE and exact propagation. It compares
one network on E11.5--P11 against two networks split at the prespecified P0.
It does not search alternative boundaries or calibrate rho for publication.
Only fully KKT-converged fits produce a model-selection curve.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from ctcpgrn.global_sparse_fit import GlobalTransitions, fit_global_sparse_stage
from run_global_three_stage_sparse import load_stage_transitions


def merge_transitions(first: GlobalTransitions,
                      second: GlobalTransitions) -> GlobalTransitions:
    return GlobalTransitions(
        np.concatenate([first.x, second.x]),
        np.concatenate([first.y, second.y]),
        np.concatenate([first.dt, second.dt]),
        np.concatenate([first.weights, second.weights]),
    )


def old_fit_as_initial(directory: Path, name: str, rows: np.ndarray,
                       cols: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = sparse.load_npz(directory / f"{name}_generator.npz").tocsr()
    baseline = np.load(directory / f"{name}_baseline.npy", allow_pickle=False)
    if matrix.shape != (p, p) or baseline.shape != (p,):
        raise ValueError(f"Stored {name} matrix does not match the common panel")
    return np.r_[np.asarray(matrix[rows, cols]).ravel(), matrix.diagonal()], baseline


def save_fit(directory: Path, name: str, fit, rows: np.ndarray,
             cols: np.ndarray, p: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    matrix = sparse.coo_matrix(
        (fit.edge_values,
         (np.r_[rows, np.arange(p)], np.r_[cols, np.arange(p)])),
        shape=(p, p),
    ).tocsr()
    sparse.save_npz(directory / f"{name}_generator.npz", matrix)
    np.save(directory / f"{name}_baseline.npy", fit.baseline)
    pd.DataFrame(fit.trace).to_csv(directory / f"{name}_trace.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path,
                        default=Path("data/processed/three_stage_global_500_30tf_allpairs_20261008"))
    parser.add_argument("--expression", type=Path,
                        default=Path("data/raw/gse74243/GSE74243_matrix_RMAlog2expressionvalues.txt.gz"))
    parser.add_argument("--prenatal-dir", type=Path,
                        default=Path("results/global_three_stage_500_30tf_prenatal_active_strict_20261007"))
    parser.add_argument("--early-dir", type=Path,
                        default=Path("results/global_three_stage_500_30tf_early_active_strict_20261007"))
    parser.add_argument("--whole-initial-dir", type=Path,
                        help="Continue the whole-period fit from a saved whole_generator.npz and whole_baseline.npy")
    parser.add_argument("--output-dir", type=Path,
                        help="Results directory; defaults depend on candidate mode")
    parser.add_argument("--penalty", type=float, default=0.15)
    parser.add_argument("--ridge", type=float, default=0.0,
                        help="Nonnegative L2 coefficient on off-diagonal edges; zero reproduces the original L1 fit")
    parser.add_argument("--candidate-mode", choices=("screened_union", "all_tf_targets"),
                        default="all_tf_targets",
                        help="Default: all TF-to-state-gene pairs; screened_union is for legacy reproduction only")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Save and audit the candidate mask without fitting networks")
    parser.add_argument("--analysis-rho", type=float, default=0.15)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--active-rounds", type=int, default=10)
    parser.add_argument("--active-add-per-round", type=int, default=100)
    parser.add_argument("--kkt-tolerance", type=float, default=1e-3)
    parser.add_argument("--objective-tolerance", type=float, default=1e-12)
    parser.add_argument("--only", choices=("all", "prenatal", "early_postnatal", "whole"),
                        default="all")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path(
            "results/real_p0_all_tf_targets_20261008" if args.candidate_mode == "all_tf_targets"
            else "results/real_p0_rho_profile_common_candidates_20261007"
        )
    prior_config = args.output_dir / "profile_configuration.json"
    if prior_config.exists():
        old_mode = json.loads(prior_config.read_text()).get("candidate_mode", "screened_union")
        if old_mode != args.candidate_mode:
            raise ValueError(f"Existing output uses candidate mode {old_mode}; choose a new --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    genes, fixed = load_stage_transitions(args.candidate_dir, args.expression)
    p = len(genes)
    parts = {"prenatal": fixed["prenatal"],
             "early_postnatal": fixed["early_postnatal"]}
    parts["whole"] = merge_transitions(parts["prenatal"], parts["early_postnatal"])
    if args.candidate_mode == "screened_union":
        candidates = pd.read_csv(args.candidate_dir / "candidate_tf_target_pairs.tsv.gz", sep="\t")
        candidates = candidates.loc[candidates.stage.isin(["prenatal", "early_postnatal"]),
                                    ["regulator", "target"]].drop_duplicates()
        candidate_rule = "Union of prenatal and early-postnatal correlation-screened pairs"
    else:
        regulators = pd.read_csv(args.candidate_dir / "selected_regulators.tsv", sep="\t")["Gene Symbol"].astype(str).tolist()
        if len(regulators) != 30 or len(set(regulators)) != 30:
            raise ValueError("Expected exactly 30 distinct selected regulators")
        if not set(regulators).issubset(set(genes)):
            raise ValueError("All selected regulators must be in the state-gene panel")
        candidates = pd.DataFrame(
            [(regulator, target) for target in genes for regulator in regulators
             if regulator != target],
            columns=["regulator", "target"],
        )
        candidate_rule = "All 30 selected TFs to all 515 state genes, excluding self-edges; no pairwise correlation screen"
    index = {gene: i for i, gene in enumerate(genes)}
    candidates["row"] = candidates.target.map(index)
    candidates["col"] = candidates.regulator.map(index)
    if candidates[["row", "col"]].isna().any().any():
        raise ValueError("Candidate names not found in gene panel")
    candidates = candidates.sort_values(["row", "col"])
    rows = candidates.row.to_numpy(dtype=np.int64)
    cols = candidates.col.to_numpy(dtype=np.int64)
    candidates.to_csv(args.output_dir / "common_candidate_edges.tsv.gz", sep="\t",
                      index=False, compression="gzip")
    metadata = {
        "comparison": "Fixed P0 versus no change, E11.5--P11 only",
        "model": "Shared global affine ODE x'=A_k x+b_k, sparse matrix-exponential action",
        "n_state_genes": p,
        "n_common_candidate_edges": len(rows),
        "candidate_rule": candidate_rule,
        "candidate_mode": args.candidate_mode,
        "alpha": args.penalty,
        "ridge": args.ridge,
        "whole_initial_dir": str(args.whole_initial_dir) if args.whole_initial_dir else None,
        "rho_is_posthoc_sensitivity_not_calibrated": True,
        "normalisation": "Same 207-sample, strain-specific standardisation and endpoint weights for all three fits",
    }
    (args.output_dir / "profile_configuration.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"candidate_mode={args.candidate_mode} n_candidates={len(rows)}", flush=True)
    if args.prepare_only:
        return
    starts = {}
    for name, directory in (("prenatal", args.prenatal_dir),
                            ("early_postnatal", args.early_dir)):
        if (directory / f"{name}_generator.npz").exists() and (directory / f"{name}_baseline.npy").exists():
            starts[name] = old_fit_as_initial(directory, name, rows, cols, p)
        elif args.candidate_mode == "all_tf_targets":
            starts[name] = None  # Recompute the Euler-LASSO initialisation from the training transitions.
        else:
            raise FileNotFoundError(f"Missing original warm start in {directory}")
    if starts["prenatal"] is not None and starts["early_postnatal"] is not None:
        count = [len(parts["prenatal"].dt), len(parts["early_postnatal"].dt)]
        starts["whole"] = (
            (count[0] * starts["prenatal"][0] + count[1] * starts["early_postnatal"][0]) / sum(count),
            (count[0] * starts["prenatal"][1] + count[1] * starts["early_postnatal"][1]) / sum(count),
        )
    else:
        starts["whole"] = None
    if args.whole_initial_dir is not None:
        starts["whole"] = old_fit_as_initial(args.whole_initial_dir, "whole", rows, cols, p)
    chosen = list(parts) if args.only == "all" else [args.only]
    for name in chosen:
        diagnostics_path = args.output_dir / f"{name}_diagnostics.tsv"
        if diagnostics_path.exists():
            print(f"[{name}] diagnostic already exists; refusing to replace", flush=True)
            continue
        table = parts[name]
        print(f"[{name}] n_transitions={len(table.dt)} n_candidates={len(rows)}", flush=True)
        started = time.perf_counter()

        def checkpoint(edge_values, intercept, entry, stage=name):
            intermediate = args.output_dir / "checkpoints" / stage / f"round_{entry['round']:02d}"
            intermediate.mkdir(parents=True, exist_ok=True)
            mat = sparse.coo_matrix(
                (edge_values,
                 (np.r_[rows, np.arange(p)], np.r_[cols, np.arange(p)])),
                shape=(p, p),
            ).tocsr()
            sparse.save_npz(intermediate / "generator.npz", mat)
            np.save(intermediate / "baseline.npy", intercept)
            pd.DataFrame([entry]).to_csv(intermediate / "diagnostics.tsv",
                                         sep="\t", index=False)
            print(f"[{stage}] round={entry['round']} objective={entry['objective']:.6f} "
                  f"kkt={entry['kkt_inf']:.6g} edges={entry['nonzero_offdiag']}", flush=True)

        fit = fit_global_sparse_stage(
            table, rows, cols, args.penalty, max_iter=args.max_iter,
            ridge=args.ridge,
            kkt_tolerance=args.kkt_tolerance,
            objective_tolerance=args.objective_tolerance,
            initial=starts[name], optimizer="active_lbfgsb",
            active_rounds=args.active_rounds,
            active_add_per_round=args.active_add_per_round,
            progress_callback=checkpoint,
        )
        elapsed = time.perf_counter() - started
        save_fit(args.output_dir, name, fit, rows, cols, p)
        diagnostic_row = {
            "stage": name, "n_transitions": len(table.dt),
            "objective_per_transition": fit.objective,
            "additive_cost": len(table.dt) * fit.objective,
            "data_loss_per_transition": fit.data_loss,
            "ridge": args.ridge,
            "n_selected_edges": fit.n_nonzero_offdiag,
            "iterations": fit.iterations, "kkt_inf": fit.kkt_inf,
            "converged": fit.converged, "runtime_seconds": elapsed,
        }
        pd.DataFrame([diagnostic_row]).to_csv(diagnostics_path, sep="\t", index=False)
        if name != "whole" and fit.converged:
            stage_dir = args.output_dir / "stage_networks" / name
            save_fit(stage_dir, name, fit, rows, cols, p)
            pd.DataFrame([diagnostic_row]).to_csv(stage_dir / "fit_diagnostics.tsv",
                                                    sep="\t", index=False)
            selected = np.abs(fit.edge_values[:len(rows)]) > 1e-8
            pd.DataFrame({
                "stage": name,
                "regulator": np.asarray(genes, dtype=str)[cols[selected]],
                "target": np.asarray(genes, dtype=str)[rows[selected]],
                "edge_weight": fit.edge_values[:len(rows)][selected],
            }).to_csv(stage_dir / "selected_stage_edges.tsv.gz", sep="\t",
                      index=False, compression="gzip")
            (stage_dir / "run_configuration.json").write_text(json.dumps({
                "candidate_dir": str(args.candidate_dir), "stage": name,
                "candidate_universe": candidate_rule,
                "penalty": args.penalty, "optimizer": "active_lbfgsb",
                "ridge": args.ridge,
                "kkt_tolerance": args.kkt_tolerance,
                "fixed_boundary": "P0",
            }, indent=2) + "\n")
        print(f"[{name}] converged={fit.converged} kkt={fit.kkt_inf:.6g} "
              f"seconds={elapsed:.1f}", flush=True)

    paths = [args.output_dir / f"{name}_diagnostics.tsv" for name in parts]
    if not all(path.exists() for path in paths):
        print("Three fits are required before the rho curve can be drawn.", flush=True)
        return
    fitted = pd.concat([pd.read_csv(path, sep="\t") for path in paths], ignore_index=True)
    if not fitted.converged.all():
        print("At least one KKT check failed. No model-selection curve reported.", flush=True)
        return
    score = fitted.set_index("stage").additive_cost
    l0 = float(score["whole"])
    split = float(score["prenatal"] + score["early_postnatal"])
    if l0 <= 0:
        raise ValueError("Relative rho requires a positive no-change cost")
    critical = (l0 - split) / l0
    rho = np.linspace(0.0, 0.9, 181)
    frame = pd.DataFrame({"rho": rho, "no_change_cost": l0,
                          "p0_split_cost": split + rho * l0})
    frame["selected_model"] = np.where(frame.p0_split_cost < l0, "P0_split", "no_change")
    frame.to_csv(args.output_dir / "rho_profile.tsv", sep="\t", index=False)
    analysed_rho = args.analysis_rho
    analysed_split_cost = split + analysed_rho * l0
    (args.output_dir / "rho_summary.json").write_text(json.dumps({
        "L0": l0, "unpenalised_P0_split_cost": split,
        "rho_critical": critical,
        "analysed_rho": analysed_rho,
        "analysed_P0_split_cost": analysed_split_cost,
        "analysed_cost_advantage_of_split": l0 - analysed_split_cost,
        "selected_at_analysed_rho": "P0_split" if analysed_split_cost < l0 else "no_change",
        "interpretation": "P0 split wins for 0 <= rho < rho_critical only if rho_critical > 0. Boundary prespecified; not a change-point scan or independent calibration."
    }, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(rho, frame.no_change_cost, color="#335C81", lw=2, label="No change")
    ax.plot(rho, frame.p0_split_cost, color="#D97942", lw=2,
            label="Prespecified P0 split + rho penalty")
    if 0 <= critical <= 0.9:
        ax.axvline(critical, color="#555555", lw=1, ls="--",
                   label=f"Critical rho = {critical:.3f}")
    ax.axvline(analysed_rho, color="#2E8B57", lw=1.2, ls=":",
               label=f"Analysed rho = {analysed_rho:.2f}")
    ax.scatter([analysed_rho], [analysed_split_cost], color="#D97942", s=38, zorder=4)
    ax.set_xlim(0, 0.35)
    ax.set(xlabel=r"Change-point penalty ratio $\rho$", ylabel="Additive model cost",
           title="Fixed-P0 model-selection sensitivity")
    ax.legend(frameon=False)
    fig.savefig(args.output_dir / "rho_profile.pdf")
    fig.savefig(args.output_dir / "rho_profile.png", dpi=220)
    plt.close(fig)
    print(f"L0={l0:.6f} P0_split={split:.6f} rho_critical={critical:.6f}", flush=True)


if __name__ == "__main__":
    main()
