#!/usr/bin/env python3
"""Recreate the 60 long E6 systems for native whole-series tree GRN baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ctcpgrn.io import read_dream_signed_network, signed_edge_list_to_matrix
from ctcpgrn.paths import results_root
from ctcpgrn.simulation import (make_irregular_times, parameterise_signed_network,
                                perturb_network, simulate_piecewise_linear_ode)
from run_revision_comparison import GOLD


ROOT = results_root()
INPUT = ROOT / "revision_e6" / "input"
OUT = ROOT / "tree_grn_long"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    signs, genes = signed_edge_list_to_matrix(read_dream_signed_network(GOLD))
    manifest = pd.read_csv(INPUT / "manifest.tsv", sep="\t")
    selected = manifest[(manifest.split == "e4test") & manifest["T"].isin([25, 41])]
    assert len(selected) == 60
    rows = []
    for row in selected.itertuples(index=False):
        seed, n_times = int(row.seed), int(row.T)
        rng = np.random.default_rng(seed)
        first = parameterise_signed_network(signs, rng, magnitude_range=(0.12, 0.45), stability_margin=0.30)
        times = make_irregular_times(n_times, float(n_times - 1), 0.50, seed + 100)
        if row.changed:
            second = perturb_network(first, rng, rewiring_fraction=0.35,
                                     sign_flip_fraction=0.10, weight_scale=0.35)
            true_index = int(rng.choice(np.arange(5, n_times - 5)))
            networks, change_times = [first, second], [float(times[true_index])]
        else:
            second, true_index = first.copy(), -1
            networks, change_times = [first], []
        simulation = simulate_piecewise_linear_ode(networks, change_times, times,
                                                    n_replicates=5, noise_std=0.03,
                                                    seed=seed + 200)
        prior_truth = np.load(INPUT / f"{row.system_id}_truth.npz")
        prior_mean = pd.read_csv(INPUT / f"{row.system_id}.tsv", sep="\t")
        assert (true_index == int(row.true_cp_index)) if row.changed else pd.isna(row.true_cp_index)
        assert np.allclose(prior_truth["A1"], first) and np.allclose(prior_truth["A2"], second)
        assert np.allclose(prior_mean.time.to_numpy(), times)
        assert np.allclose(prior_mean.iloc[:, 1:].to_numpy(), simulation.observations.mean(axis=0))
        np.savez_compressed(OUT / f"system_{seed}.npz", observations=simulation.observations,
                            first=first, second=second, times=times, genes=np.asarray(genes),
                            true_index=true_index)
        rows.append({"seed": seed, "T": n_times, "changed": bool(row.changed),
                     "true_index": true_index, "system_id": row.system_id})
    pd.DataFrame(rows).to_csv(OUT / "input_manifest.tsv", sep="\t", index=False)
    print(f"Prepared {len(rows)} verified long-series systems in {OUT}")


if __name__ == "__main__":
    main()
