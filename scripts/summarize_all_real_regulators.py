#!/usr/bin/env python3
"""Display every candidate regulator in the two converged GSE74243 stages.

Reads the existing fitted networks. A signed target-expression score is made
only when a stage/regulator has at least one selected non-regulator target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from analyze_two_converged_stage_networks import (
    STAGE_DIRS,
    STRAIN_MARKERS,
    calculate_regulon_proxy,
    load_converged_edges,
    load_sample_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path,
                        default=Path("data/processed/three_stage_global_500_30tf_20261007"))
    parser.add_argument("--expression", type=Path,
                        default=Path("data/raw/gse74243/GSE74243_matrix_RMAlog2expressionvalues.txt.gz"))
    parser.add_argument("--prenatal-dir", type=Path, default=STAGE_DIRS["prenatal"])
    parser.add_argument("--early-dir", type=Path, default=STAGE_DIRS["early_postnatal"])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/real_all_regulators_v3_20261007"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to replace existing results: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    edges = load_converged_edges({"prenatal": args.prenatal_dir,
                                  "early_postnatal": args.early_dir})
    candidates = pd.read_csv(args.candidate_dir / "selected_regulators.tsv", sep="\t")
    regulators = candidates["Gene Symbol"].astype(str).tolist()
    assert len(regulators) == len(set(regulators)) == 30
    regulator_set = set(regulators)

    rows = []
    for regulator in regulators:
        row = {"regulator": regulator, "candidate_rank": regulators.index(regulator) + 1}
        for stage in ("prenatal", "early_postnatal"):
            part = edges[stage].loc[edges[stage].regulator.eq(regulator)]
            nonreg = part.loc[~part.target.isin(regulator_set)]
            row[f"{stage}_all_edges"] = len(part)
            row[f"{stage}_nonreg_targets"] = len(nonreg)
            row[f"{stage}_signature_defined"] = bool(len(nonreg))
        rows.append(row)
    coverage = pd.DataFrame(rows)
    coverage["total_edges"] = (coverage.prenatal_all_edges
                               + coverage.early_postnatal_all_edges)
    coverage = coverage.sort_values(["total_edges", "candidate_rank"],
                                    ascending=[False, True]).reset_index(drop=True)
    coverage.to_csv(args.output_dir / "candidate_regulator_coverage.tsv", sep="\t", index=False)
    chosen = [(stage, str(row.regulator))
              for stage in ("prenatal", "early_postnatal")
              for row in coverage.sort_values(
                  [f"{stage}_all_edges", "candidate_rank"],
                  ascending=[False, True]).itertuples(index=False)
              if getattr(row, f"{stage}_nonreg_targets") > 0]

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    count_cols = ["prenatal_all_edges", "prenatal_nonreg_targets",
                  "early_postnatal_all_edges", "early_postnatal_nonreg_targets"]
    counts = coverage[count_cols].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(8.1, 10.6), constrained_layout=True)
    image = ax.imshow(np.log1p(counts), aspect="auto", cmap="Blues", vmin=0,
                      vmax=np.log1p(counts.max()))
    ax.set_xticks(range(4), ["E: all edges", "E: non-candidate targets",
                             "P: all edges", "P: non-candidate targets"], rotation=25,
                  ha="right")
    ax.set_yticks(range(len(coverage)), coverage.regulator)
    ax.set_title("All 30 candidate regulators in two converged stage networks")
    ax.axvline(1.5, color="white", linewidth=2)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center",
                    fontsize=8, color="white" if counts[i, j] >= 40 else "#222222")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02,
                 label="log(1 + selected-edge count)")
    fig.savefig(args.output_dir / "candidate_regulator_coverage.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "candidate_regulator_coverage.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)

    samples, metadata, _ = load_sample_panel(args.expression, args.candidate_dir)
    activity, signatures = calculate_regulon_proxy(
        samples, metadata, edges, chosen, args.candidate_dir)
    activity.to_csv(args.output_dir / "all_defined_regulon_scores.tsv.gz", sep="\t",
                    index=False, compression="gzip")
    signatures.to_csv(args.output_dir / "all_defined_regulon_signatures.tsv", sep="\t",
                      index=False)
    grouped = activity.groupby(["regulon_label", "network_stage", "strain"],
                               observed=True).signed_target_expression_score.mean()
    groups = [(stage, strain) for stage in ("prenatal", "early_postnatal")
              for strain in STRAIN_MARKERS]
    labels = [f"{regulator} ({'E' if stage == 'prenatal' else 'P'})"
              for stage, regulator in chosen]
    matrix = np.array([[grouped.loc[(label, stage, strain)]
                        for stage, strain in groups] for label in labels])
    means = pd.DataFrame(matrix, index=labels,
                         columns=[f"{stage}_{strain}" for stage, strain in groups])
    means.index.name = "regulon_label"
    means.to_csv(args.output_dir / "all_defined_regulon_group_means.tsv", sep="\t")

    limit = max(0.15, float(np.quantile(np.abs(matrix), 0.95)))
    fig, ax = plt.subplots(figsize=(8.1, max(4.0, 0.48 * len(labels) + 1.8)),
                           constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                      norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    ax.set_xticks(range(len(groups)),
                  [f"{'E' if stage == 'prenatal' else 'P'} · {strain}"
                   for stage, strain in groups])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Every defined stage-specific signed target-expression signature")
    ax.axvline(2.5, color="white", linewidth=2)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.025,
                 label="Mean signed target-expression z-score")
    fig.savefig(args.output_dir / "all_defined_regulon_proxy.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "all_defined_regulon_proxy.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)

    summary = {"candidate_regulators": len(regulators),
               "with_any_selected_edge": int((coverage.total_edges > 0).sum()),
               "without_selected_edges": int((coverage.total_edges == 0).sum()),
               "active_prenatal": int((coverage.prenatal_all_edges > 0).sum()),
               "active_early_postnatal": int((coverage.early_postnatal_all_edges > 0).sum()),
               "defined_stage_regulator_signatures": len(chosen),
               "signature_rule": "At least one non-regulator target in that stage; up to 20 strongest absolute-weight targets; same signed z-score rule as main figure",
               "main_figure_selection": "Up to two largest non-regulator-target outdegrees independently within each stage"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
