#!/usr/bin/env python3
"""Prepare frozen long-sequence simulations for E6 external baselines.

EDISON and VARDetect accept one multivariate series.  We therefore expose the
replicate mean (not a concatenation across replicate boundaries) and retain the
continuous-time truth for topology scoring in Python.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.io import read_dream_signed_network, signed_edge_list_to_matrix
from ctcpgrn.paths import results_root
from ctcpgrn.simulation import (
    make_irregular_times,
    parameterise_signed_network,
    perturb_network,
    simulate_piecewise_linear_ode,
)


GOLD = Path("data/raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv")
OUT = results_root() / "revision_e6/input"


def make_system(seed: int, n_times: int, changed: bool, split: str) -> dict:
    signs, _ = signed_edge_list_to_matrix(read_dream_signed_network(GOLD))
    rng = np.random.default_rng(seed)
    a1 = parameterise_signed_network(
        signs, rng, magnitude_range=(0.12, 0.45), stability_margin=0.30
    )
    times = make_irregular_times(n_times, float(n_times - 1), 0.50, seed + 100)
    if changed:
        a2 = perturb_network(
            a1,
            rng,
            rewiring_fraction=0.35,
            sign_flip_fraction=0.10,
            weight_scale=0.35,
        )
        cp_index = int(rng.choice(np.arange(5, n_times - 5)))
        networks, cps = [a1, a2], [float(times[cp_index])]
    else:
        a2, cp_index, networks, cps = a1.copy(), None, [a1], []
    sim = simulate_piecewise_linear_ode(
        networks,
        cps,
        times,
        n_replicates=5,
        noise_std=0.03,
        seed=seed + 200,
    )
    system_id = f"{split}_T{n_times}_{'change' if changed else 'null'}_{seed}"
    mean_obs = sim.observations.mean(axis=0)
    frame = pd.DataFrame(mean_obs, columns=[f"G{i + 1}" for i in range(mean_obs.shape[1])])
    frame.insert(0, "time", times)
    frame.to_csv(OUT / f"{system_id}.tsv", sep="\t", index=False)
    np.savez_compressed(OUT / f"{system_id}_truth.npz", A1=a1, A2=a2)
    return {
        "system_id": system_id,
        "split": split,
        "seed": seed,
        "T": n_times,
        "changed": changed,
        "true_cp_index": cp_index,
        "true_cp_time": None if cp_index is None else float(times[cp_index]),
        "mean_dt": float(np.mean(np.diff(times))),
        "irregularity_sd_log_dt": float(np.std(np.log(np.diff(times)))),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, n_per_class, base in (("calibration", 4, 610000), ("test", 15, 620000)):
        for ti, n_times in enumerate((25, 41)):
            for changed in (False, True):
                offset = 10000 * ti + (1000 if changed else 0)
                for j in range(n_per_class):
                    rows.append(make_system(base + offset + j + 1, n_times, changed, split))
    # Exact seeds used in E4's strong-change cells, enabling a paired external
    # comparison without re-running CT--CP--GRN.
    for n_times, null_base, change_base in ((25, 140000, 152200), (41, 150000, 162200)):
        for j in range(1, 16):
            rows.append(make_system(null_base + j, n_times, False, "e4test"))
            rows.append(make_system(change_base + j, n_times, True, "e4test"))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT / "manifest.tsv", sep="\t", index=False)
    with (OUT / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "topology": "DREAM4 size-10 network 1 signed topology",
                "replicates": 5,
                "external_input": "replicate mean; gene-wise z-score inside R runner",
                "change_strength": {"rewiring_fraction": 0.35, "weight_scale": 0.35},
                "sampling_irregularity": 0.50,
                "noise_std": 0.03,
                "calibration_systems": 16,
                "test_systems": 60,
                "paired_e4_test_systems": 60,
            },
            handle,
            indent=2,
        )
    print(manifest.groupby(["split", "T", "changed"]).size())


if __name__ == "__main__":
    main()
