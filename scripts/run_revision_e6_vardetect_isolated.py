#!/usr/bin/env python3
"""Run VARDetect one system per R process so native-code failures are auditable."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from ctcpgrn.paths import results_root


ROOT = Path(__file__).resolve().parents[1]
RESULTS = results_root() / "revision_e6"
R_DEPENDENCY_CHECK = (
    'pkgs <- c("VARDetect", "sparsevar"); '
    'missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)]; '
    'if (length(missing)) stop(paste("Missing R package(s):", '
    'paste(missing, collapse=", ")), call.=FALSE)'
)


def _candidate_rscripts() -> list[str]:
    """Return unique Rscript candidates, respecting an explicit override."""
    explicit = os.environ.get("CTCPGRN_RSCRIPT")
    if explicit:
        resolved = shutil.which(explicit)
        explicit_path = Path(resolved or explicit).expanduser()
        if explicit_path.is_file() and os.access(explicit_path, os.X_OK):
            return [str(explicit_path.resolve())]
        return []

    candidates: list[str | None] = [shutil.which("Rscript")]

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        candidates.append(str(Path(conda_exe).resolve().with_name("Rscript")))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        active_prefix = Path(conda_prefix).resolve()
        candidates.append(str(active_prefix / "bin" / "Rscript"))
        if active_prefix.parent.name == "envs":
            candidates.append(str(active_prefix.parent.parent / "bin" / "Rscript"))

    prefix = Path(sys.prefix).resolve()
    candidates.append(str(prefix / "bin" / "Rscript"))
    if prefix.parent.name == "envs":
        candidates.append(str(prefix.parent.parent / "bin" / "Rscript"))

    unique: list[str] = []
    for candidate in candidates:
        normalized = str(Path(candidate).expanduser().resolve()) if candidate else None
        if (
            normalized
            and normalized not in unique
            and Path(normalized).is_file()
            and os.access(normalized, os.X_OK)
        ):
            unique.append(normalized)
    return unique


def _select_rscript() -> str:
    """Select the first Rscript whose library contains the required packages."""
    candidates = _candidate_rscripts()
    if not candidates:
        explicit = os.environ.get("CTCPGRN_RSCRIPT")
        prefix = (
            f"CTCPGRN_RSCRIPT='{explicit}' is not an executable file; fix it or unset it to enable automatic discovery. "
            if explicit else ""
        )
        raise SystemExit(
            prefix + "Cannot find a usable Rscript. Set CTCPGRN_RSCRIPT to the absolute path of "
            "the Rscript that contains VARDetect and sparsevar."
        )

    failures: list[str] = []
    for candidate in candidates:
        try:
            dependency_check = subprocess.run(
                [candidate, "-e", R_DEPENDENCY_CHECK],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"- {candidate}: dependency preflight timed out after 30 seconds")
            continue
        except OSError as exc:
            failures.append(f"- {candidate}: cannot execute ({exc})")
            continue
        if dependency_check.returncode == 0:
            print(f"Using Rscript: {candidate}", flush=True)
            return candidate
        details = dependency_check.stderr.strip() or dependency_check.stdout.strip()
        details = " ".join(details.split())[-500:]
        failures.append(f"- {candidate}: {details}")

    explicit = os.environ.get("CTCPGRN_RSCRIPT")
    hint = (
        "The explicit CTCPGRN_RSCRIPT failed dependency validation."
        if explicit
        else "No discovered Rscript passed dependency validation."
    )
    raise SystemExit(
        f"{hint}\nTried:\n" + "\n".join(failures) +
        "\nInstall VARDetect and sparsevar in one of these R environments, or "
        "set CTCPGRN_RSCRIPT to a usable Rscript."
    )


def main() -> None:
    if sys.argv[1:] == ["--check-r"]:
        _select_rscript()
        print("VARDetect R dependency preflight OK", flush=True)
        return
    split = sys.argv[1] if len(sys.argv) > 1 else "calibration"
    rscript = _select_rscript()
    manifest = pd.read_csv(RESULTS / "input" / "manifest.tsv", sep="\t")
    systems = manifest.loc[manifest["split"] == split, "system_id"].tolist()
    summaries, edges, failures = [], [], []
    for i, system_id in enumerate(systems, start=1):
        command = [
            rscript, "scripts/run_revision_e6_external.R",
            split, "vardetect", system_id,
        ]
        try:
            completed = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, timeout=120
            )
            timeout_message = ""
        except subprocess.TimeoutExpired as exc:
            completed = None
            stderr = exc.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            timeout_message = f"timeout after 120 seconds: {stderr[-400:]}"
        stem = f"external_{split}_vardetect_{system_id}"
        summary_path = RESULTS / f"{stem}_summary.tsv"
        edge_path = RESULTS / f"{stem}_edges.tsv"
        if (
            completed is not None
            and completed.returncode == 0
            and summary_path.exists()
            and edge_path.exists()
        ):
            summaries.append(pd.read_csv(summary_path, sep="\t"))
            edges.append(pd.read_csv(edge_path, sep="\t"))
            status = "ok"
        else:
            status = "timeout" if completed is None else "native_failure"
            failures.append({
                "system_id": system_id,
                "returncode": 124 if completed is None else completed.returncode,
                "stderr_tail": (
                    timeout_message if completed is None
                    else completed.stderr[-500:].replace("\n", " ")
                ),
            })
        print(f"{i}/{len(systems)} {system_id}: {status}", flush=True)
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(
            RESULTS / f"external_{split}_vardetect_summary.tsv", sep="\t", index=False
        )
    if edges:
        pd.concat(edges, ignore_index=True).to_csv(
            RESULTS / f"external_{split}_vardetect_edges.tsv", sep="\t", index=False
        )
    pd.DataFrame(failures, columns=["system_id", "returncode", "stderr_tail"]).to_csv(
        RESULTS / f"external_{split}_vardetect_failures.tsv", sep="\t", index=False
    )


if __name__ == "__main__":
    main()
