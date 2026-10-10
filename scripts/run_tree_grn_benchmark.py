#!/usr/bin/env python3
"""Native whole-series Arboreto GENIE3/GRNBoost2 benchmark.

Run with an environment containing arboreto. Inputs are
prepared separately by prepare_tree_grn_long.py with the simulation
environment. Both tree methods receive every observation from the same five
replicate trajectories and each return one network per system.
The official per-target Arboreto core is called sequentially, avoiding the
Dask network scheduler while preserving the packaged RF/GBM configurations.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from arboreto.core import (EARLY_STOP_WINDOW_LENGTH, RF_KWARGS,
                           SGBM_KWARGS, infer_partial_network)
from sklearn.metrics import average_precision_score


def fit_stage(method: str, values: np.ndarray, genes: list[str], seed: int):
    frame = pd.DataFrame(values.reshape(-1, len(genes)), columns=genes)
    regressor, kwargs = ("RF", RF_KWARGS) if method == "genie3" else ("GBM", SGBM_KWARGS)
    x = frame.to_numpy()
    links = pd.concat([
        infer_partial_network(regressor, kwargs, x, genes, target, x[:, index],
                              early_stop_window_length=EARLY_STOP_WINDOW_LENGTH, seed=seed)
        for index, target in enumerate(genes)
    ], ignore_index=True)
    assert {"TF", "target", "importance"}.issubset(links.columns)
    position = {gene: index for index, gene in enumerate(genes)}
    scores = np.zeros((len(genes), len(genes)), dtype=float)
    for row in links.itertuples(index=False):
        scores[position[row.target], position[row.TF]] = row.importance
    assert np.isfinite(scores).all()
    np.fill_diagonal(scores, 0.0)
    return scores, links


def score(truth: np.ndarray, estimate: np.ndarray):
    mask = ~np.eye(len(truth), dtype=bool)
    return float(average_precision_score((truth[mask] != 0).astype(int), estimate[mask]))


def run_one(root: Path, seed: int, method: str) -> dict:
    label = f"{method}_global"
    path = root / f"system_{seed}_{label}.tsv"
    if path.is_file():
        rows = pd.read_csv(path, sep="\t")
        assert set(rows.regime) == {1, 2} and len(rows) == 2
        return rows.iloc[0].to_dict() | {"network_aupr": float(rows.stage_aupr.mean())}
    data = np.load(root / f"system_{seed}.npz")
    observations = data["observations"]
    first, second = data["first"], data["second"]
    genes = data["genes"].tolist()
    matrix, _ = fit_stage(method, observations, genes, seed * 10)
    rows = []
    for regime, truth in enumerate((first, second), 1):
        rows.append({"seed": seed, "method": label, "regime": regime,
                     "boundary_index": -1, "n_observations": int(np.prod(observations.shape[:2])),
                     "stage_aupr": score(truth, matrix)})
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    np.savez_compressed(root / f"system_{seed}_{label}_scores.npz", before=matrix, after=matrix)
    return rows[0] | {"network_aupr": float(np.mean([row["stage_aupr"] for row in rows]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*")
    args = parser.parse_args()
    root = args.root.resolve()
    assert (root / "input_manifest.tsv").is_file()
    manifest = pd.read_csv(root / "input_manifest.tsv", sep="\t")
    seeds = args.seeds if args.seeds is not None else manifest.seed.astype(int).tolist()
    assert seeds and set(seeds).issubset(set(manifest.seed.astype(int)))
    for seed in seeds:
        for method in ("genie3", "grnboost2"):
            result = run_one(root, seed, method)
            print(f"{seed} {method} global AP={result['network_aupr']:.6f}", flush=True)
    (root / "arboreto_versions.tsv").write_text(
        "package\tversion\n" + "\n".join(f"{name}\t{version(name)}" for name in
                                ("arboreto", "scikit-learn", "distributed", "dask", "numpy", "pandas")) + "\n"
    )


if __name__ == "__main__":
    main()
