#!/usr/bin/env python3
"""Frozen DiSCGRN/Euler tests on independent DREAM4 size-10 topologies.

Networks 2--5 are original DREAM4 signed gold-standard topologies. They are
not the synthetic rewiring controls derived from network 1. Their expression
series here are newly simulated affine systems, not DREAM4 challenge samples.
Network 1 is excluded because it supplied the development topology.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import scan_one_change_candidates, select_one_change
from ctcpgrn.experiment_params import load_frozen_params
from ctcpgrn.io import read_dream_signed_network, signed_edge_list_to_matrix, sha256sum
from ctcpgrn.paths import results_root
from run_optimized_benchmark import make_dataset, score_result, trajectory_groups


GOLD_ROOT = Path("data/raw/dream4/goldstandards")


def one_system(task: tuple[int, int, bool, str, dict]) -> list[dict]:
    network, seed, changed, gold_string, params = task
    gold = Path(gold_string)
    _, first, second, true_index, simulation = make_dataset(gold, seed, changed)
    rows = []
    for method in ("ct", "euler"):
        alpha, rho, budget = params[method]
        path = scan_one_change_candidates(
            trajectory_groups(simulation), simulation.times, method, alpha,
            minimum_transitions=5, max_iter=budget,
            ct_initialiser="euler" if method == "ct" else "dplsq",
        )
        chosen = select_one_change(path, rho, relative_beta=True)
        true_time = None if true_index is None else float(simulation.times[true_index])
        rows.append({
            "network": network, "seed": seed, "changed": changed,
            "method": method, "alpha": alpha, "rho": rho, "max_iter": budget,
            "true_change_time": true_time, "selected_change_time": chosen.change_time,
            **score_result(chosen, true_time, first, second),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-parameters", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=results_root() / "independent_dream_topologies")
    parser.add_argument("--n-per-condition", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.n_per_condition < 1 or args.workers < 1:
        raise ValueError("n-per-condition and workers must be positive")
    frozen = load_frozen_params(args.selected_parameters)
    golds = {index: GOLD_ROOT / f"insilico_size10_{index}_goldstandard_signed.tsv"
             for index in range(2, 6)}
    for index, path in golds.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing independent network {index}: {path}")
        signs, genes = signed_edge_list_to_matrix(read_dream_signed_network(path))
        if signs.shape != (10, 10) or len(genes) != 10:
            raise ValueError(f"Invalid 10-gene topology: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "goldstandards": {str(i): {"path": str(path), "sha256": sha256sum(path)}
                          for i, path in golds.items()},
        "development_topology": "DREAM4 size-10 network 1 (excluded)",
        "n_per_condition": args.n_per_condition, "networks": [2, 3, 4, 5],
        "changed_and_stationary": True,
        "generator": "make_dataset: piecewise affine ODE, not original DREAM4 expression data",
        "frozen_parameters": {key: list(frozen[key]) for key in ("ct", "euler")},
        "seed_rule": "800000 + 10000*network + 100*int(changed) + replicate_index",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    tasks = []
    for index, gold in golds.items():
        for changed in (True, False):
            for replicate in range(1, args.n_per_condition + 1):
                seed = 800000 + 10000 * index + 100 * int(changed) + replicate
                tasks.append((index, seed, changed, str(gold), frozen))
    if args.workers == 1:
        blocks = [one_system(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            blocks = list(pool.map(one_system, tasks))
    frame = pd.DataFrame(row for block in blocks for row in block)
    frame.to_csv(args.output_dir / "system_results.tsv", sep="\t", index=False)
    records = []
    for index in golds:
        item = frame[frame.network.eq(index)]
        changed = item[item.changed]
        stationary = item[~item.changed]
        ct = changed[changed.method.eq("ct")].sort_values("seed")
        euler = changed[changed.method.eq("euler")].sort_values("seed")
        if not np.array_equal(ct.seed.to_numpy(), euler.seed.to_numpy()):
            raise RuntimeError(f"Paired seeds differ for network {index}")
        delta = ct.network_aupr.to_numpy() - euler.network_aupr.to_numpy()
        rng = np.random.default_rng(20261006 + index)
        bootstrap = rng.choice(delta, size=(5000, len(delta)), replace=True).mean(axis=1)
        record = {"network": index, "n_changed": len(delta),
                  "n_stationary": int(stationary[stationary.method.eq("ct")].shape[0]),
                  "ct_network_aupr": float(ct.network_aupr.mean()),
                  "euler_network_aupr": float(euler.network_aupr.mean()),
                  "paired_aupr_difference": float(delta.mean()),
                  "paired_ci_lower": float(np.quantile(bootstrap, 0.025)),
                  "paired_ci_upper": float(np.quantile(bootstrap, 0.975))}
        for method in ("ct", "euler"):
            pos = changed[changed.method.eq(method)]
            neg = stationary[stationary.method.eq(method)]
            record[f"{method}_change_match"] = float(pos.cp_f1.mean())
            record[f"{method}_stationary_correct"] = float(neg.cp_f1.mean())
        records.append(record)
    summary = pd.DataFrame(records)
    summary.to_csv(args.output_dir / "topology_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
