#!/usr/bin/env python3
"""Summarise and bootstrap the paired joint-parameter development grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.paths import results_root


def variance_shares(flat_scores, shape):
    y = np.asarray(flat_scores).reshape(shape)
    grand = y.mean()
    alpha = y.mean(axis=(1, 2), keepdims=True) - grand
    budget = y.mean(axis=(0, 2), keepdims=True) - grand
    rho = y.mean(axis=(0, 1), keepdims=True) - grand
    alpha_budget = y.mean(axis=2, keepdims=True) - grand - alpha - budget
    alpha_rho = y.mean(axis=1, keepdims=True) - grand - alpha - rho
    budget_rho = y.mean(axis=0, keepdims=True) - grand - budget - rho
    triple = y - grand - alpha - budget - rho - alpha_budget - alpha_rho - budget_rho
    parts = {
        "alpha": alpha, "H": budget, "rho": rho,
        "alpha:H": alpha_budget, "alpha:rho": alpha_rho,
        "H:rho": budget_rho, "alpha:H:rho": triple,
    }
    total = np.square(y - grand).sum()
    return {
        name: float(np.square(np.broadcast_to(part, y.shape)).sum() / total)
        for name, part in parts.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2068)
    args = parser.parse_args()
    out = results_root() / "joint_parameter_sensitivity"
    systems = pd.read_csv(out / "system_scores.tsv", sep="\t")
    grid = pd.read_csv(out / "grid.tsv", sep="\t")
    keys = ["alpha", "max_iter", "beta_ratio"]
    columns = systems[keys].drop_duplicates().sort_values(keys)
    shape = tuple(columns[key].nunique() for key in keys)
    if shape != (3, 4, 19) or len(grid) != np.prod(shape):
        raise RuntimeError("Expected a complete 3 x 4 x 19 grid")
    changed = systems[systems.changed]
    stationary = systems[~systems.changed]
    if changed.seed.nunique() != 15 or stationary.seed.nunique() != 15:
        raise RuntimeError("Expected 15 changed and 15 stationary systems")
    wide = {}
    for key, frame, metric in (
        ("cp", changed, "cp_f1"),
        ("change", changed, "change_aupr"),
        ("network", changed, "network_aupr"),
        ("specificity", stationary, "cp_f1"),
    ):
        pivot = frame.pivot(index="seed", columns=keys, values=metric)
        wide[key] = pivot.sort_index(axis=1).to_numpy()
        if wide[key].shape != (15, 228) or not np.isfinite(wide[key]).all():
            raise RuntimeError(f"Incomplete or non-finite {key} matrix")
    rng = np.random.default_rng(args.seed)
    records = []
    rho_levels = sorted(columns.beta_ratio.unique())
    local_rho = [index for index, rho in enumerate(rho_levels) if 0.20 <= rho <= 0.45]
    for replicate in range(args.bootstrap + 1):
        if replicate == 0:
            changed_idx = stationary_idx = np.arange(15)
        else:
            changed_idx = rng.integers(0, 15, 15)
            stationary_idx = rng.integers(0, 15, 15)
        score = (
            wide["cp"][changed_idx].mean(axis=0)
            + 0.35 * wide["change"][changed_idx].mean(axis=0)
            + 0.20 * wide["network"][changed_idx].mean(axis=0)
            + 0.50 * wide["specificity"][stationary_idx].mean(axis=0)
        )
        records.append({"replicate": replicate, "scope": "full_0_to_0_90",
                        **variance_shares(score, shape)})
        local_score = score.reshape(shape)[:, :, local_rho].ravel()
        records.append({"replicate": replicate, "scope": "local_0_20_to_0_45",
                        **variance_shares(local_score, (3, 4, len(local_rho)))})
    boot = pd.DataFrame(records)
    boot.to_csv(out / "interaction_bootstrap_replicates.tsv", sep="\t", index=False)
    summary = []
    for scope, scoped in boot.groupby("scope", sort=False):
        for component in scoped.columns.drop(["replicate", "scope"]):
            values = scoped.loc[scoped.replicate > 0, component].to_numpy()
            summary.append({
                "scope": scope, "component": component,
                "variance_share": scoped.loc[scoped.replicate == 0, component].iloc[0],
                "bootstrap_2_5pct": np.quantile(values, 0.025),
                "bootstrap_97_5pct": np.quantile(values, 0.975),
            })
    pd.DataFrame(summary).to_csv(out / "interaction_bootstrap_summary.tsv", sep="\t", index=False)
    best = (grid.sort_values(
        ["alpha", "max_iter", "selection_score", "mean_rmse", "beta_ratio"],
        ascending=[True, True, False, True, True], kind="stable",
    ).drop_duplicates(["alpha", "max_iter"])
        .sort_values(["alpha", "max_iter"]))
    best.to_csv(out / "best_rho_by_alpha_H.tsv", sep="\t", index=False)
    print("paired systems:", changed.seed.nunique(), "changed +", stationary.seed.nunique(), "stationary")
    print("grid cells:", len(grid), "bootstrap replicates:", args.bootstrap)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
