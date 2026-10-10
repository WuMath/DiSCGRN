#!/usr/bin/env python3
"""Run one Arboreto baseline in its own Python environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_tree_grn_benchmark import fit_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("genie3", "grnboost2"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    binned = pd.read_csv(args.input, index_col=0)
    genes = binned.columns.astype(str).tolist()
    scores, _ = fit_stage(args.method, binned.to_numpy(dtype=float), genes, args.seed)
    pd.DataFrame(scores, index=genes, columns=genes).to_csv(args.output)


if __name__ == "__main__":
    main()
