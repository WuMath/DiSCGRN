"""Independent regulatory evidence from CollecTRI and ChIP-Atlas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_collectri(path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", low_memory=False)
    required = {"source_genesymbol", "target_genesymbol", "is_stimulation", "is_inhibition"}
    if not required.issubset(table.columns):
        raise ValueError(f"CollecTRI table lacks {sorted(required - set(table.columns))}")
    result = table.copy()
    result["regulator"] = result["source_genesymbol"].astype(str)
    result["target"] = result["target_genesymbol"].astype(str)
    stimulation = result["is_stimulation"].astype(str).str.lower().eq("true")
    inhibition = result["is_inhibition"].astype(str).str.lower().eq("true")
    result["sign"] = np.select([stimulation & ~inhibition, inhibition & ~stimulation], [1, -1], default=0)
    return result[["regulator", "target", "sign", "sources", "references"]].drop_duplicates()


def collectri_panel_edges(table: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    universe = set(genes)
    return table[table["regulator"].isin(universe) & table["target"].isin(universe)].copy()


def available_chipatlas_antigens(index_path: str | Path, genome: str = "mm10") -> set[str]:
    payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if genome not in payload:
        raise KeyError(f"Genome {genome} not found in ChIP-Atlas index")
    return set(payload[genome])


def read_chipatlas_target_table(path: str | Path) -> pd.DataFrame:
    """Convert one wide ChIP-Atlas target table to gene-level evidence."""

    path = Path(path)
    table = pd.read_csv(path, sep="\t", low_memory=False)
    if table.shape[1] < 2:
        raise ValueError(f"Unexpected ChIP-Atlas format: {path}")
    target_column, average_column = table.columns[:2]
    experiment_columns = list(table.columns[2:])
    lung_columns = [column for column in experiment_columns if "lung" in column.lower()]
    result = pd.DataFrame(
        {
            "regulator": path.name.split(".")[0],
            "target": table[target_column].astype(str),
            "average_score": pd.to_numeric(table[average_column], errors="coerce").fillna(0.0),
        }
    )
    if lung_columns:
        lung_values = table[lung_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        result["lung_score"] = lung_values.max(axis=1)
        result["has_lung_experiment"] = True
    else:
        result["lung_score"] = np.nan
        result["has_lung_experiment"] = False
    result["cross_tissue_support"] = result["average_score"] > 0
    result["lung_support"] = result["lung_score"].fillna(0) > 0
    return result


def read_chipatlas_directory(path: str | Path) -> pd.DataFrame:
    frames = [read_chipatlas_target_table(item) for item in sorted(Path(path).glob("*.tsv"))]
    if not frames:
        return pd.DataFrame(columns=["regulator", "target", "average_score", "lung_score", "has_lung_experiment", "cross_tissue_support", "lung_support"])
    return pd.concat(frames, ignore_index=True)


def validation_metrics(
    inferred: np.ndarray,
    genes: list[str],
    evidence: pd.DataFrame,
) -> dict[str, float]:
    """Score all evaluable TF-target pairs, not only predicted positives."""

    index = {gene: i for i, gene in enumerate(genes)}
    regulators = sorted(set(evidence["regulator"]) & set(genes))
    if not regulators:
        return {"n_evaluable_pairs": 0, "evidence_aupr": float("nan"), "sign_concordance": float("nan")}
    positives = {(row.regulator, row.target): int(row.sign) for row in evidence.itertuples(index=False)}
    labels, scores, concordant = [], [], []
    for regulator in regulators:
        for target in genes:
            if regulator == target:
                continue
            labels.append(int((regulator, target) in positives))
            value = inferred[index[target], index[regulator]]
            scores.append(abs(value))
            sign = positives.get((regulator, target), 0)
            if sign != 0:
                concordant.append(int(np.sign(value) == sign))
    from sklearn.metrics import average_precision_score

    return {
        "n_evaluable_pairs": len(labels),
        "evidence_aupr": float(average_precision_score(labels, scores)),
        "sign_concordance": float(np.mean(concordant)) if concordant else float("nan"),
    }
