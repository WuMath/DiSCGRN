#!/usr/bin/env python3
"""Run the authors' SCODE.R on frozen DREAM simulations and matched BEELINE bins.

SCODE yields one directed network per input. On nonstationary DREAM systems the
same score matrix is evaluated against both true stages, then their APs are
averaged. On BEELINE, the shared static directed gold standard is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from run_beeline_matched_benchmark import score_network


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DREAM = PROJECT / "results/runs/weighted_score_20261004/tree_grn_long"
DEFAULT_BEELINE = PROJECT / "results/beeline_extended_valid_20261009"
DEFAULT_OUTPUT = PROJECT / "results/runs/grn_accuracy_20261009/scode"
SOURCE = PROJECT / "third_party/SCODE/SCODE.R"
WRAPPER = PROJECT / "scripts/run_official_scode_seeded.R"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_metrics(truth: np.ndarray, scores: np.ndarray) -> dict:
    mask = ~np.eye(len(truth), dtype=bool)
    y = (truth[mask] != 0).astype(int)
    q = scores[mask]
    if y.min() == y.max() or not np.isfinite(q).all():
        raise ValueError("Nonfinite scores or only one edge class")
    return {"aupr_ap": float(average_precision_score(y, q)),
            "auroc": float(roc_auc_score(y, q))}


def fit_scode(values: np.ndarray, times: np.ndarray, output: Path,
              *, trials: int, iterations: int, latent_dim: int, rscript: str,
              seed_base: int) -> tuple[np.ndarray, list[float]]:
    """values are observations x genes and times have one entry per row."""
    n_obs, p = values.shape
    if n_obs != len(times) or p < 3 or not np.isfinite(values).all():
        raise ValueError("Invalid SCODE input array")
    if latent_dim >= p:
        raise ValueError("SCODE latent dimension must be below gene count")
    output.mkdir(parents=True, exist_ok=True)
    scaled = (times - times.min()) / (times.max() - times.min())
    expression_file = output / "expression_for_scode.tsv"
    time_file = output / "time_for_scode.tsv"
    np.savetxt(expression_file, values.T, delimiter="\t", fmt="%.12g")
    np.savetxt(time_file, np.column_stack([np.arange(n_obs), scaled]),
               delimiter="\t", fmt=["%d", "%.12g"])
    matrices = []
    durations = []
    for trial in range(trials):
        trial_dir = output / f"trial_{trial + 1:02d}"
        matrix_file = trial_dir / "A.txt"
        if not matrix_file.is_file():
            env = os.environ.copy()
            env.update({"SCODE_SOURCE": str(SOURCE), "SCODE_SEED": str(seed_base + trial),
                        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            cmd = [rscript, str(WRAPPER), str(expression_file), str(time_file),
                   str(trial_dir), str(p), str(latent_dim), str(n_obs), str(iterations)]
            start = time.perf_counter()
            process = subprocess.run(cmd, cwd=PROJECT, env=env, text=True,
                                     capture_output=True, timeout=600)
            durations.append(time.perf_counter() - start)
            (output / f"trial_{trial + 1:02d}.log").write_text(
                process.stdout + "\n" + process.stderr)
            if process.returncode:
                raise RuntimeError(f"SCODE trial {trial + 1} exited {process.returncode}")
        else:
            durations.append(0.0)
        matrix = np.loadtxt(matrix_file, delimiter="\t")
        if matrix.shape != (p, p) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid SCODE network: {matrix_file}")
        matrices.append(matrix)
    average = np.mean(matrices, axis=0)
    scores = np.abs(average)
    np.fill_diagonal(scores, 0.0)
    return scores, durations


def run_dream(path: Path, output: Path, settings: argparse.Namespace) -> dict:
    seed = int(path.stem.removeprefix("system_"))
    data = np.load(path)
    observations = data["observations"]
    n_reps, n_times, p = observations.shape
    times = data["times"]
    values = observations.reshape(n_reps * n_times, p)
    repeated_times = np.tile(times, n_reps)
    scores, durations = fit_scode(values, repeated_times, output,
                                  trials=settings.trials, iterations=settings.iterations,
                                  latent_dim=settings.latent_dim, rscript=settings.rscript,
                                  seed_base=seed * 10)
    first = stage_metrics(data["first"], scores)
    second = stage_metrics(data["second"], scores)
    np.savez_compressed(output / "scores.npz", scores=scores)
    return {"dataset": "DREAM_simulated", "run_id": f"system_{seed}",
            "T": n_times, "changed": bool(data["true_index"] >= 0),
            "n_genes": p, "aupr_ap": (first["aupr_ap"] + second["aupr_ap"]) / 2,
            "auroc": (first["auroc"] + second["auroc"]) / 2,
            "stage1_aupr": first["aupr_ap"], "stage2_aupr": second["aupr_ap"],
            "runtime_seconds": sum(durations), "status": "ok"}


def run_beeline(path: Path, output: Path, settings: argparse.Namespace) -> dict:
    label = path.parent.parent.name
    run_id = path.parent.name
    frame = pd.read_csv(path, index_col=0)
    genes = frame.columns.astype(str).tolist()
    truth = pd.read_csv(PROJECT / "data/raw/beeline/Synthetic" /
                        label.rsplit("-", 1)[0] / run_id / "GroundTruth.csv")
    scores, durations = fit_scode(frame.to_numpy(float), frame.index.to_numpy(float),
                                  output, trials=settings.trials,
                                  iterations=settings.iterations,
                                  latent_dim=settings.latent_dim, rscript=settings.rscript,
                                  seed_base=20261009 + 100 * int(run_id.rsplit("-", 1)[-1])
                                  + sum(ord(char) for char in label))
    metrics = score_network(scores, genes, truth)
    pd.DataFrame(scores, index=genes, columns=genes).to_csv(output / "SCODE_scores.csv")
    return {"dataset": label, "run_id": run_id, "n_genes": len(genes),
            **metrics, "runtime_seconds": sum(durations), "status": "ok"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("dream", "beeline", "all"), default="all")
    parser.add_argument("--dream-root", type=Path, default=DEFAULT_DREAM)
    parser.add_argument("--beeline-root", type=Path, default=DEFAULT_BEELINE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--rscript", default="/usr/local/bin/Rscript")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.source.resolve() != SOURCE.resolve():
        raise ValueError("Use the audited official SCODE source in third_party/SCODE")
    if args.trials < 1 or args.iterations < 2:
        raise ValueError("Invalid SCODE trial or iteration count")
    jobs = []
    if args.domain in ("dream", "all"):
        for path in sorted(args.dream_root.glob("system_[0-9]*.npz")):
            if not re.fullmatch(r"system_\d+\.npz", path.name):
                continue
            jobs.append(("dream", path, args.output_root / "dream" / path.stem))
    if args.domain in ("beeline", "all"):
        for path in sorted(args.beeline_root.glob("dyn-*/dyn-*-*/binned_input.csv")):
            jobs.append(("beeline", path, args.output_root / "beeline" /
                         path.parent.parent.name / path.parent.name))
    if args.limit:
        jobs = jobs[:args.limit]
    if not jobs:
        raise FileNotFoundError("No frozen DREAM or BEELINE inputs found")
    args.output_root.mkdir(parents=True, exist_ok=True)
    protocol = {"source_url": "https://github.com/hmatsu1226/SCODE",
                "source_sha256": sha256(SOURCE), "method": "official SCODE.R",
                "trials": args.trials, "iterations": args.iterations,
                "latent_dim": args.latent_dim,
                "dream_input": "all five trajectories stacked as columns; true times retained",
                "beeline_input": "same 13-bin pseudotime means as existing benchmark",
                "score": "absolute mean signed A over trials, directed non-self AP"}
    protocol_file = args.output_root / "protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError("Existing output uses a different SCODE protocol")
    protocol_file.write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    for domain, path, output in jobs:
        metric_file = output / "SCODE_metrics.json"
        if metric_file.is_file():
            row = json.loads(metric_file.read_text())
        else:
            try:
                row = run_dream(path, output, args) if domain == "dream" else run_beeline(path, output, args)
            except Exception as error:
                row = {"dataset": domain, "run_id": output.name, "status": "failed",
                       "error": f"{type(error).__name__}: {error}"}
            output.mkdir(parents=True, exist_ok=True)
            metric_file.write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
        print(row["run_id"], row["status"], row.get("aupr_ap", row.get("error")), flush=True)
    pd.DataFrame(rows).to_csv(args.output_root / "all_metrics.tsv", sep="\t", index=False)
    print(f"Completed {sum(row['status'] == 'ok' for row in rows)}/{len(rows)} jobs")


if __name__ == "__main__":
    main()
