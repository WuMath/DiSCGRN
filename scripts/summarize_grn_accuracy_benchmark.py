#!/usr/bin/env python3
"""Combine frozen GRN edge-ranking results with official SCODE/dynGENIE3 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "results/runs/grn_accuracy_20261009"
FROZEN = PROJECT / "results/runs/weighted_score_20261004"
BEELINE = PROJECT / "results/beeline_extended_valid_20261009/summary/all_metrics.tsv"
METHODS = ["DiSCGRN", "Euler--DP", "EDISON", "GENIE3", "GRNBoost2", "dynGENIE3", "SCODE"]


def dream_table(root: Path) -> pd.DataFrame:
    manifest = pd.read_csv(FROZEN / "tree_grn_long/input_manifest.tsv", sep="\t")
    meta = manifest.set_index("seed")
    entries = []
    frozen = pd.read_csv(FROZEN / "revision_e6/e6_all_systems.tsv", sep="\t")
    for row in frozen.itertuples(index=False):
        if row.method not in ("DiSCGRN", "Euler--DP", "EDISON"):
            continue
        entries.append({"domain": "DREAM_simulated", "dataset": f"T{int(row.T)}",
                        "run_id": f"system_{int(row.seed)}", "changed": bool(row.changed),
                        "method": row.method, "aupr_ap": row.network_aupr,
                        "status": row.status, "runtime_seconds": row.runtime_seconds})
    for path in sorted((FROZEN / "tree_grn_long").glob("system_*_global.tsv")):
        frame = pd.read_csv(path, sep="\t")
        if len(frame) != 2 or set(frame.regime) != {1, 2}:
            raise ValueError(f"Expected two stage scores: {path}")
        seed = int(frame.seed.iloc[0])
        method = {"genie3_global": "GENIE3", "grnboost2_global": "GRNBoost2"}.get(
            frame.method.iloc[0])
        if method is None:
            continue
        entries.append({"domain": "DREAM_simulated", "dataset": f"T{int(meta.loc[seed, 'T'])}",
                        "run_id": f"system_{seed}", "changed": bool(meta.loc[seed, "changed"]),
                        "method": method, "aupr_ap": frame.stage_aupr.mean(), "status": "ok"})
    dyn = pd.read_csv(FROZEN / "dyngenie3_long/system_aupr.tsv", sep="\t")
    for row in dyn.itertuples(index=False):
        entries.append({"domain": "DREAM_simulated", "dataset": f"T{int(row.T)}",
                        "run_id": f"system_{int(row.seed)}", "changed": bool(row.changed),
                        "method": "dynGENIE3", "aupr_ap": row.network_aupr,
                        "status": row.status, "runtime_seconds": row.runtime_seconds})
    scode = pd.read_csv(root / "scode/all_metrics.tsv", sep="\t")
    for row in scode[scode.dataset == "DREAM_simulated"].itertuples(index=False):
        seed = int(row.run_id.removeprefix("system_"))
        entries.append({"domain": "DREAM_simulated", "dataset": f"T{int(meta.loc[seed, 'T'])}",
                        "run_id": row.run_id, "changed": bool(meta.loc[seed, "changed"]),
                        "method": "SCODE", "aupr_ap": row.aupr_ap,
                        "status": row.status, "runtime_seconds": row.runtime_seconds})
    return pd.DataFrame(entries)


def beeline_table(root: Path) -> pd.DataFrame:
    existing = pd.read_csv(BEELINE, sep="\t")
    cols = ["dataset", "run_id", "method", "aupr_ap", "auroc", "precision_at_k",
            "max_f1", "status", "runtime_seconds"]
    entries = [existing[cols].copy()]
    scode = pd.read_csv(root / "scode/all_metrics.tsv", sep="\t")
    scode = scode[scode.dataset != "DREAM_simulated"].copy()
    scode["method"] = "SCODE"
    entries.append(scode[cols])
    dyn = pd.read_csv(root / "dyngenie3_beeline/all_metrics.tsv", sep="\t")
    dyn["method"] = "dynGENIE3"
    entries.append(dyn[cols])
    combined = pd.concat(entries, ignore_index=True)
    combined.insert(0, "domain", "BEELINE")
    return combined


def bootstrap_paired(group: pd.DataFrame, reference: str, competitor: str,
                     seed: int, draws: int) -> dict:
    matrix = group.pivot(index="run_id", columns="method", values="aupr_ap")
    paired = matrix[[reference, competitor]].dropna()
    differences = (paired[reference] - paired[competitor]).to_numpy()
    rng = np.random.default_rng(seed)
    replicates = rng.choice(differences, size=(draws, len(differences)), replace=True).mean(axis=1)
    lo, hi = np.quantile(replicates, [0.025, 0.975])
    return {"reference": reference, "competitor": competitor,
            "n_paired": len(differences), "reference_mean": paired[reference].mean(),
            "competitor_mean": paired[competitor].mean(),
            "mean_ap_difference": differences.mean(), "ci_low": lo, "ci_high": hi}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    output = args.root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([dream_table(args.root), beeline_table(args.root)], ignore_index=True)
    if combined.duplicated(["domain", "dataset", "run_id", "method"]).any():
        raise ValueError("Duplicate run-method rows")
    combined.to_csv(output / "all_run_metrics.tsv", sep="\t", index=False)
    ok = combined[(combined.status == "ok") & np.isfinite(combined.aupr_ap)].copy()
    summary = ok.groupby(["domain", "dataset", "method"], as_index=False).agg(
        n_evaluable=("aupr_ap", "size"), mean_aupr=("aupr_ap", "mean"),
        sd_aupr=("aupr_ap", "std"), mean_auroc=("auroc", "mean"),
        mean_precision_at_k=("precision_at_k", "mean"))
    planned = combined.groupby(["domain", "dataset", "method"]).size().rename("n_planned")
    summary = summary.join(planned, on=["domain", "dataset", "method"])
    summary.to_csv(output / "dataset_summary.tsv", sep="\t", index=False)
    paired_rows = []
    for (domain, dataset), group in ok.groupby(["domain", "dataset"]):
        for competitor in METHODS:
            if competitor == "DiSCGRN" or competitor not in set(group.method):
                continue
            row = bootstrap_paired(group, "DiSCGRN", competitor,
                                   seed=20261009 + sum(map(ord, dataset + competitor)),
                                   draws=args.bootstrap)
            paired_rows.append({"domain": domain, "dataset": dataset, **row})
    pd.DataFrame(paired_rows).to_csv(output / "paired_aupr.tsv", sep="\t", index=False)
    checks = {"dream_systems": int(combined[combined.domain == "DREAM_simulated"].run_id.nunique()),
              "beeline_runs": int(combined[combined.domain == "BEELINE"].run_id.nunique()),
              "failed_rows": int((combined.status != "ok").sum()),
              "n_rows": int(len(combined)),
              "bootstrap_draws": args.bootstrap,
              "note": "DREAM static methods: each full-series score is evaluated against both true stages and their APs averaged. BEELINE uses one static truth per run."}
    (output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
