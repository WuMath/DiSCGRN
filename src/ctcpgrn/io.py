"""Readers and preprocessing helpers for the two study datasets."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLE_RE = re.compile(r"^(AJ|B6|C3H)_(e\d+(?:\.\d+)?|P\d+)_", re.IGNORECASE)


@dataclass
class LongitudinalGroup:
    """Mean observations and uncertainty for one strain/condition."""

    name: str
    times: np.ndarray
    values: np.ndarray
    weights: np.ndarray
    n_replicates: np.ndarray


def sha256sum(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def developmental_day(label: str) -> float:
    """Map mouse stages to a monotone day scale.

    Embryonic stages retain their E-day value. P0 is placed at day 20.0, a
    documented numerical convention that keeps E19.5 and P0 distinct; Pn is
    then day 20+n. This avoids a zero time interval in the ODE likelihood.
    """

    if label.lower().startswith("e"):
        return float(label[1:])
    if label.upper().startswith("P"):
        return 20.0 + float(label[1:])
    raise ValueError(f"Unrecognised developmental stage: {label}")


def read_gse74243_expression(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the GEO gene-level RMA matrix and parse its sample columns."""

    table = pd.read_csv(path, sep="\t", compression="infer", low_memory=False)
    required = {"Gene ID", "Gene Symbol"}
    if not required.issubset(table.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(table.columns))}")

    records: list[dict[str, object]] = []
    sample_columns: list[str] = []
    for column in table.columns[2:]:
        match = SAMPLE_RE.match(str(column))
        if match is None:
            continue
        strain, stage = match.groups()
        records.append(
            {
                "sample": column,
                "strain": strain.upper(),
                "stage": stage,
                "time": developmental_day(stage),
            }
        )
        sample_columns.append(column)

    if len(sample_columns) != 216:
        raise ValueError(f"Expected 216 samples, parsed {len(sample_columns)}")
    metadata = pd.DataFrame.from_records(records).sort_values(["strain", "time", "sample"])
    expression = table[["Gene ID", "Gene Symbol", *sample_columns]].copy()
    expression[sample_columns] = expression[sample_columns].apply(pd.to_numeric, errors="coerce")
    return expression, metadata.reset_index(drop=True)


def collapse_duplicate_symbols(expression: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicated gene symbols using the row with highest variance."""

    sample_columns = list(expression.columns[2:])
    work = expression.dropna(subset=["Gene Symbol"]).copy()
    work = work[work["Gene Symbol"].astype(str).str.len() > 0]
    work["_variance"] = work[sample_columns].var(axis=1, ddof=1)
    best = work.sort_values("_variance", ascending=False).drop_duplicates("Gene Symbol")
    return best.drop(columns="_variance").reset_index(drop=True)


def select_variable_genes(expression: pd.DataFrame, n_genes: int) -> pd.DataFrame:
    sample_columns = list(expression.columns[2:])
    variance = expression[sample_columns].var(axis=1, ddof=1)
    return expression.loc[variance.nlargest(n_genes).index].reset_index(drop=True)


def build_gse74243_groups(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    variance_floor_quantile: float = 0.25,
) -> list[LongitudinalGroup]:
    """Aggregate replicates and compute inverse-variance observation weights."""

    genes = expression["Gene Symbol"].astype(str).to_numpy()
    groups: list[LongitudinalGroup] = []
    for strain, strain_meta in metadata.groupby("strain", sort=True):
        means, variances, ns, times = [], [], [], []
        for time, time_meta in strain_meta.groupby("time", sort=True):
            cols = time_meta["sample"].tolist()
            values = expression[cols].to_numpy(dtype=float).T
            means.append(np.nanmean(values, axis=0))
            variances.append(np.nanvar(values, axis=0, ddof=1) if len(cols) > 1 else np.full(len(genes), np.nan))
            ns.append(len(cols))
            times.append(float(time))
        means_arr = np.asarray(means)
        var_arr = np.asarray(variances)
        finite = var_arr[np.isfinite(var_arr) & (var_arr > 0)]
        floor = np.quantile(finite, variance_floor_quantile) if finite.size else 1.0
        var_arr = np.where(np.isfinite(var_arr), np.maximum(var_arr, floor), floor)
        n_arr = np.asarray(ns, dtype=float)
        weights = n_arr[:, None] / var_arr
        weights /= np.nanmean(weights)
        groups.append(
            LongitudinalGroup(
                name=strain,
                times=np.asarray(times),
                values=means_arr,
                weights=weights,
                n_replicates=n_arr,
            )
        )
    return groups


def read_dream_signed_network(path: str | Path) -> pd.DataFrame:
    """Read a DREAM4 signed gold-standard edge list."""

    frame = pd.read_csv(path, sep="\t", header=None, comment="#")
    if frame.shape[1] < 3:
        raise ValueError(f"Expected at least three columns in {path}")
    frame = frame.iloc[:, :3]
    frame.columns = ["source", "target", "sign"]
    sign_text = frame["sign"].astype(str).str.strip()
    mapped = sign_text.map({"+": 1, "-": -1, "1": 1, "-1": -1, "0": 0})
    if mapped.isna().any():
        raise ValueError(f"Unknown DREAM sign values: {sorted(sign_text[mapped.isna()].unique())}")
    frame["sign"] = mapped.astype(int)
    return frame


def signed_edge_list_to_matrix(edges: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    genes = sorted(set(edges["source"]) | set(edges["target"]), key=lambda x: int(re.sub(r"\D", "", str(x))))
    index = {gene: i for i, gene in enumerate(genes)}
    matrix = np.zeros((len(genes), len(genes)), dtype=float)
    for row in edges.itertuples(index=False):
        matrix[index[row.target], index[row.source]] = np.sign(row.sign)
    return matrix, genes


def write_manifest(paths: list[str | Path], output: str | Path, root: str | Path) -> None:
    root = Path(root).resolve()
    records = []
    for item in paths:
        path = Path(item).resolve()
        records.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256sum(path),
            }
        )
    Path(output).write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
