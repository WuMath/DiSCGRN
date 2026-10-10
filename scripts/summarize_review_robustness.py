#!/usr/bin/env python3
"""Paired bootstrap for the review robustness experiment's changed systems."""

from pathlib import Path

import numpy as np
import pandas as pd
from ctcpgrn.paths import results_root


ROOT = results_root() / "robustness"


def main():
    frame = pd.read_csv(ROOT / "robustness_systems.tsv", sep="\t")
    frame = frame[frame.changed]
    rng = np.random.default_rng(20261003)
    rows = []
    for (topology, generator), group in frame.groupby(["topology", "generator"]):
        wide = group.pivot(index="seed", columns="method", values="network_aupr")
        delta = (wide.ct - wide.euler).to_numpy()
        draws = delta[rng.integers(len(delta), size=(20000, len(delta)))].mean(axis=1)
        rows.append({"topology": topology, "generator": generator,
                     "n_changed": len(delta), "ct_minus_euler": delta.mean(),
                     "ci_low": np.quantile(draws, 0.025),
                     "ci_high": np.quantile(draws, 0.975)})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "robustness_paired.tsv", sep="\t", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
