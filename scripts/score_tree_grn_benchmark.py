#!/usr/bin/env python3
"""Pair stage-wise tree GRN baselines with the frozen DiSCGRN test results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ctcpgrn.paths import results_root


ROOT = results_root()
OUT = ROOT / "tree_grn_benchmark"
SEEDS = set(range(99601, 99616))


def paired_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 20_000):
    draw = rng.integers(0, len(values), (n_boot, len(values)))
    return tuple(map(float, (np.mean(values), *np.quantile(values[draw].mean(axis=1), [0.025, 0.975]))))


def main():
    original = pd.read_csv(ROOT / "revision_comparison_h80" / "comparison_30systems.tsv", sep="\t")
    ct = original[(original.changed) & (original.method == "ct")][["seed", "network_aupr"]]
    assert set(ct.seed) == SEEDS and len(ct) == 15
    rows = []
    for seed in sorted(SEEDS):
        for method in ("genie3", "grnboost2", "genie3_oracle", "grnboost2_oracle"):
            path = OUT / f"system_{seed}_{method}.tsv"
            if not path.is_file():
                raise FileNotFoundError(path)
            stages = pd.read_csv(path, sep="\t")
            assert len(stages) == 2 and list(stages.regime) == [1, 2]
            assert stages.seed.eq(seed).all() and stages.method.eq(method).all()
            rows.append({"seed": seed, "method": method,
                         "network_aupr": float(stages.stage_aupr.mean()),
                         "boundary_index": int(stages.boundary_index.iloc[0])})
    system = pd.concat([pd.DataFrame(rows), ct.assign(method="discgrn", boundary_index=np.nan)], ignore_index=True)
    assert system.groupby("method").size().eq(15).all()
    system.to_csv(OUT / "system_aupr.tsv", sep="\t", index=False)
    wide = system.pivot(index="seed", columns="method", values="network_aupr")
    rng = np.random.default_rng(20261007)
    pairs = []
    for method in ("genie3", "grnboost2", "genie3_oracle", "grnboost2_oracle"):
        delta = (wide.discgrn - wide[method]).to_numpy()
        mean, low, high = paired_ci(delta, rng)
        pairs.append({"comparison": f"discgrn_vs_{method}", "n_systems": 15,
                      "mean_difference": mean, "ci_low": low, "ci_high": high})
    pd.DataFrame(pairs).to_csv(OUT / "paired_bootstrap.tsv", sep="\t", index=False)
    summary = system.groupby("method").network_aupr.agg(["count", "mean", "std"]).reset_index()
    summary.to_csv(OUT / "summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    print(pd.DataFrame(pairs).to_string(index=False))


if __name__ == "__main__":
    main()
