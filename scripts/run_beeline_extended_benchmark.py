#!/usr/bin/env python3
"""Run and summarize seven prespecified matched-input BEELINE series."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SERIES = (("dyn-LI", 100), ("dyn-LI", 200), ("dyn-LI", 500),
          ("dyn-LL", 100), ("dyn-LL", 200), ("dyn-LL", 500),
          ("dyn-CY", 200))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/raw/beeline"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/beeline_extended_valid_20261009"))
    parser.add_argument("--rscript", default="/opt/miniconda3/bin/Rscript")
    parser.add_argument("--tree-python", required=True,
                        help="Python environment with Arboreto for GENIE3/GRNBoost2")
    parser.add_argument("--only", nargs="*", help="Optional labels such as dyn-LI-100")
    args = parser.parse_args()
    labels = {f"{network}-{cells}" for network, cells in SERIES}
    if args.only and not set(args.only).issubset(labels):
        raise ValueError(f"Unknown series: {sorted(set(args.only) - labels)}")
    chosen = [(network, cells) for network, cells in SERIES
              if not args.only or f"{network}-{cells}" in args.only]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, ["src", "scripts", env.get("PYTHONPATH")]))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    runner = Path(__file__).with_name("run_beeline_matched_benchmark.py")
    for network, cells in chosen:
        label = f"{network}-{cells}"
        command = [sys.executable, str(runner), "--input-root", str(args.input_root),
                   "--pattern", f"Synthetic/{network}/{label}-*",
                   "--dataset-label", label, "--output-root", str(args.output_root / label),
                   "--rscript", args.rscript, "--tree-python", args.tree_python]
        print(f"Running {label}", flush=True)
        subprocess.run(command, env=env, check=True)
    summary = Path(__file__).with_name("summarize_beeline_matched_benchmark.py")
    roots = [args.output_root / f"{network}-{cells}" for network, cells in SERIES]
    if all((root / "all_metrics.tsv").is_file() for root in roots):
        subprocess.run([sys.executable, str(summary), "--roots", *map(str, roots),
                        "--output-root", str(args.output_root / "summary")], env=env, check=True)
    else:
        print("Series subset complete; summary deferred until all seven series exist.", flush=True)


if __name__ == "__main__":
    main()
