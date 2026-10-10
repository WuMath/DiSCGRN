#!/usr/bin/env python3
"""Score frozen E6 external baselines and join the paired E4 CT/Euler results."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from ctcpgrn.experiment_params import load_frozen_params
from ctcpgrn.paths import results_root


ROOT = results_root() / "revision_e6"
INPUT = ROOT / "input"
EDISON_THRESHOLD = 0.08


def edge_matrices(edges: pd.DataFrame, system_id: str, config: str) -> list[np.ndarray]:
    part = edges[(edges.system_id == system_id) & (edges.config == config)]
    matrices = []
    for _, group in part.groupby("regime"):
        group = group.sort_values(["target", "regulator"])
        matrices.append(group.score.to_numpy().reshape(10, 10))
    return matrices


def network_aupr(system_id: str, matrices: list[np.ndarray], stationary: bool = False,
                 cp_index: float | None = None, n_times: int | None = None) -> float:
    truth = np.load(INPUT / f"{system_id}_truth.npz")
    mask = ~np.eye(10, dtype=bool)
    if stationary and len(matrices) == 2:
        weight = 0.5 if cp_index is None or n_times is None else np.clip(cp_index / (n_times - 1), 0, 1)
        pooled = weight * matrices[0] + (1 - weight) * matrices[1]
        matrices = [pooled, pooled]
    scores = []
    for matrix, key in zip(matrices[:2], ("A1", "A2")):
        if not np.all(np.isfinite(matrix)):
            return float("nan")
        y = (np.abs(truth[key]) > 0)[mask].astype(int)
        scores.append(average_precision_score(y, np.abs(matrix)[mask]))
    return float(np.mean(scores))


def parse_positions(value) -> list[int]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [int(float(item)) - 1 for item in str(value).split(";")]


def score_edison(manifest: pd.DataFrame) -> pd.DataFrame:
    summary = pd.read_csv(ROOT / "external_e4test_edison_summary.tsv", sep="\t")
    edges = pd.read_csv(ROOT / "external_e4test_edison_edges.tsv", sep="\t")
    summary = summary[summary.config == "poisson"].copy()
    summary = summary.merge(manifest, on="system_id")
    rows = []
    for row in summary.itertuples(index=False):
        selected = row.cp_score >= EDISON_THRESHOLD
        mats = edge_matrices(edges, row.system_id, row.config)
        ap = network_aupr(row.system_id, mats, not selected, row.primary_cp_index, row.T)
        error = np.nan if not row.changed or not selected else abs(row.primary_cp_index - row.true_cp_index)
        rows.append({
            "system_id": row.system_id, "seed": row.seed, "T": row.T,
            "changed": row.changed, "method": "EDISON", "applicable": True,
            "selected_change": selected, "within_one": bool(error <= 1) if row.changed else not selected,
            "index_error": error, "network_aupr": ap,
            "runtime_seconds": row.runtime_seconds, "status": row.status,
        })
    return pd.DataFrame(rows)


def score_vardetect(manifest: pd.DataFrame) -> pd.DataFrame:
    summary = pd.read_csv(ROOT / "external_e4test_vardetect_summary.tsv", sep="\t")
    edges = pd.read_csv(ROOT / "external_e4test_vardetect_edges.tsv", sep="\t")
    summary = summary.merge(manifest, on="system_id")
    rows = []
    for row in summary.itertuples(index=False):
        positions = parse_positions(row.cp_positions)
        selected = len(positions) > 0
        error = np.nan if not row.changed or not selected else min(abs(pos - row.true_cp_index) for pos in positions)
        mats = edge_matrices(edges, row.system_id, row.config)
        ap = network_aupr(row.system_id, mats, not selected, row.primary_cp_index, row.T)
        rows.append({
            "system_id": row.system_id, "seed": row.seed, "T": row.T,
            "changed": row.changed, "method": "VARDetect", "applicable": True,
            "selected_change": selected, "within_one": bool(error <= 1) if row.changed else not selected,
            "index_error": error, "network_aupr": ap,
            "runtime_seconds": row.runtime_seconds, "status": row.status,
        })
    observed = set(summary.system_id)
    for row in manifest.itertuples(index=False):
        if row.system_id not in observed:
            rows.append({
                "system_id": row.system_id, "seed": row.seed, "T": row.T,
                "changed": row.changed, "method": "VARDetect", "applicable": False,
                "selected_change": np.nan, "within_one": np.nan, "index_error": np.nan,
                "network_aupr": np.nan, "runtime_seconds": np.nan, "status": "native_failure",
            })
    return pd.DataFrame(rows)


def load_internal() -> pd.DataFrame:
    frames = []
    for n_times in (25, 41):
        changed = pd.read_csv(
            results_root() / "revision_e4_single" / f"changed_T{n_times}_r0.35_w0.35.tsv", sep="\t"
        )
        null = pd.read_csv(results_root() / "revision_e4_single" / f"null_T{n_times}.tsv", sep="\t")
        frame = pd.concat([changed, null], ignore_index=True)
        frame["applicable"] = True
        frame["runtime_seconds"] = np.nan
        frame["status"] = "ok"
        frames.append(frame[[
            "seed", "T", "changed", "method", "applicable", "selected_change",
            "within_one", "index_error", "network_aupr", "runtime_seconds", "status",
        ]])
    result = pd.concat(frames, ignore_index=True)
    result["method"] = result.method.map({"ct": "DiSCGRN", "euler": "Euler--DP"})
    result["system_id"] = result.apply(
        lambda r: f"e4test_T{int(r['T'])}_{'change' if r.changed else 'null'}_{int(r.seed)}", axis=1
    )
    return result


def bootstrap_differences(frame: pd.DataFrame, reps: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(20260930)
    rows = []
    pivot = frame.pivot_table(index=["T", "system_id"], columns="method", values="network_aupr")
    for t in (25, 41):
        part = pivot.loc[t]
        for comparator in ("Euler--DP", "EDISON", "VARDetect"):
            if comparator not in part or "DiSCGRN" not in part:
                continue
            delta = (part["DiSCGRN"] - part[comparator]).dropna().to_numpy()
            if len(delta) == 0:
                continue
            boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(reps)])
            rows.append({
                "T": t, "comparison": f"DiSCGRN - {comparator}", "n": len(delta),
                "mean_difference": delta.mean(), "ci_low": np.quantile(boot, 0.025),
                "ci_high": np.quantile(boot, 0.975),
            })
    return pd.DataFrame(rows)


def main() -> None:
    load_frozen_params()
    manifest = pd.read_csv(INPUT / "manifest.tsv", sep="\t")
    manifest = manifest[manifest.split == "e4test"].copy()
    all_results = pd.concat(
        [load_internal(), score_edison(manifest), score_vardetect(manifest)], ignore_index=True
    )
    all_results.to_csv(ROOT / "e6_all_systems.tsv", sep="\t", index=False)
    summary = all_results.groupby(["T", "method"], as_index=False).agg(
        n_total=("system_id", "size"), applicability=("applicable", "mean"),
        network_evaluable=("network_aupr", lambda x: x.notna().mean()),
        network_aupr=("network_aupr", "mean"), runtime_median_seconds=("runtime_seconds", "median"),
    )
    operating = all_results[all_results.applicable].groupby(["T", "method"])
    summary = summary.merge(operating.apply(
        lambda g: pd.Series({
            "detection_rate": g.loc[g.changed, "selected_change"].astype(float).mean(),
            "within_one_rate": g.loc[g.changed, "within_one"].astype(float).mean(),
            "specificity": 1 - g.loc[~g.changed, "selected_change"].astype(float).mean(),
            "median_index_error": g.loc[g.changed & g.selected_change.astype(bool), "index_error"].median(),
        }), include_groups=False).reset_index(), on=["T", "method"], how="left")
    summary.to_csv(ROOT / "e6_summary.tsv", sep="\t", index=False)
    bootstrap_differences(all_results).to_csv(ROOT / "e6_paired_bootstrap.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    print("\nPaired differences:\n", bootstrap_differences(all_results).to_string(index=False))


if __name__ == "__main__":
    main()
