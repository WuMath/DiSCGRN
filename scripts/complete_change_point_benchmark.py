#!/usr/bin/env python3
"""Fill missing frozen-method change-point fits on the existing test systems.

The T=13 EDISON input is prepared here for the unchanged external R runner.
The T=25/41 DPLSQ--DP and Two-stage fits use the original E4 seeds and the
development-only parameters in selected_parameters.tsv.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from ctcpgrn.baselines import select_one_change
from ctcpgrn.experiment_params import load_frozen_params
from ctcpgrn.paths import results_root
from run_optimized_benchmark import fit_path, make_dataset
from run_revision_e4_single import GOLD, _dataset


def prepare_short_edison() -> None:
    root = results_root() / "revision_e6" / "input"
    manifest_path = root / "manifest.tsv"
    manifest = pd.read_csv(manifest_path, sep="\t")
    if manifest.split.eq("frozen_short").any():
        assert manifest[manifest.split.eq("frozen_short")].shape[0] == 30
        return
    rows = []
    for changed, seeds in ((True, range(99601, 99616)), (False, range(99701, 99716))):
        for seed in seeds:
            _, _, _, index, simulation = make_dataset(GOLD, seed, changed)
            system_id = f"frozen_short_T13_{'change' if changed else 'null'}_{seed}"
            values = simulation.observations.mean(axis=0)
            frame = pd.DataFrame(values, columns=[f"G{i+1}" for i in range(values.shape[1])])
            frame.insert(0, "time", simulation.times)
            frame.to_csv(root / f"{system_id}.tsv", sep="\t", index=False)
            rows.append({"system_id": system_id, "split": "frozen_short", "seed": seed,
                         "T": 13, "changed": changed, "true_cp_index": index,
                         "true_cp_time": None if index is None else float(simulation.times[index]),
                         "mean_dt": float(simulation.times[-1] / 12),
                         "irregularity_sd_log_dt": float("nan")})
    pd.concat([manifest, pd.DataFrame(rows)], ignore_index=True).to_csv(
        manifest_path, sep="\t", index=False)
    print("prepared 30 T=13 mean sequences for EDISON", flush=True)


def _long_job(job: tuple[int, bool, int, str]) -> dict:
    seed, changed, length, method = job
    _, _, true_index, simulation = _dataset(seed, changed, length, 0.35, 0.35)
    alpha, rho, budget = load_frozen_params()[method]
    path = fit_path(method, simulation, alpha, 0.0, budget, 5)
    fit = select_one_change(path, rho, relative_beta=True)
    return {"seed": seed, "T": length, "changed": changed, "method": method,
            "alpha": alpha, "rho": rho, "H": budget, "true_index": true_index,
            "estimated_index": fit.change_index,
            "selected_change": fit.change_index is not None,
            "true_time": None if true_index is None else float(simulation.times[true_index]),
            "estimated_time": fit.change_time}


def run_long(workers: int) -> None:
    out = results_root() / "change_points" / "long_internal_missing.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(out, sep="\t") if out.exists() else pd.DataFrame()
    if not existing.empty:
        existing = existing.drop_duplicates(["seed", "changed", "T", "method"], keep="first")
        existing.to_csv(out, sep="\t", index=False)
    complete = (set(zip(existing["seed"], existing["changed"], existing["T"], existing["method"]))
                if not existing.empty else set())
    jobs = []
    for length, null_base, change_base in ((25, 140000, 152200), (41, 150000, 162200)):
        for changed, base in ((False, null_base), (True, change_base)):
            for seed in range(base + 1, base + 16):
                for method in ("dplsq", "two_stage"):
                    job = (seed, changed, length, method)
                    if job not in complete:
                        jobs.append(job)
    if not jobs:
        print("all 120 missing long fits already present", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count, row in enumerate(pool.map(_long_job, jobs), start=1):
            existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
            existing.to_csv(out, sep="\t", index=False)
            if count % 10 == 0 or count == len(jobs):
                print(f"completed {count}/{len(jobs)} missing long fits", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=("prepare-short", "long", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.part in ("prepare-short", "all"):
        prepare_short_edison()
    if args.part in ("long", "all"):
        run_long(args.workers)


if __name__ == "__main__":
    main()
