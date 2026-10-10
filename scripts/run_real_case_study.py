#!/usr/bin/env python3
"""Reproducible NKX2-1 regulator-centred case analysis for the real-data fit.

The full-data model has selected no common change point. This analysis
therefore studies context-specific single-stage networks, not inferred
temporal rewiring. It does not alter the simulation or real-data fits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ctcpgrn.real_fit import fit_method
from run_real_data_analysis import STRAINS, _bootstrap_data


def select_focus(edges: pd.DataFrame, regulator: str, n_targets: int) -> list[str]:
    """Choose targets by absolute edge weight in the strongest strain.

    The biological anchor is fixed before looking at the network. The displayed
    targets are chosen algorithmically from the output and then evaluated
    against independent ChIP and knockout measurements.
    """
    subset = edges[(edges.method == "discgrn") & (edges.stage == 1) &
                   (edges.regulator == regulator)]
    if subset.empty:
        raise ValueError(f"No inferred edges for {regulator}")
    strongest = (subset.assign(abs_weight=subset.edge_weight.abs())
                 .groupby("strain").abs_weight.sum().idxmax())
    return (subset[subset.strain.eq(strongest)]
            .assign(abs_weight=lambda frame: frame.edge_weight.abs())
            .sort_values(["abs_weight", "target"], ascending=[False, True])
            .head(n_targets).target.tolist())


def make_case_table(edges: pd.DataFrame, evidence: pd.DataFrame,
                    regulator: str, frequencies: pd.DataFrame) -> pd.DataFrame:
    subset = edges[(edges.method == "discgrn") & (edges.stage == 1) &
                   (edges.regulator == regulator)]
    weights = subset.pivot(index="target", columns="strain", values="edge_weight")
    weights = weights.rename(columns={strain: f"weight_{strain}" for strain in STRAINS})
    evidence = evidence[evidence.regulator.eq(regulator)].set_index("target")
    frequency = frequencies[frequencies.regulator.eq(regulator)]
    frequency = frequency.set_index("target")
    fields = ["rank", "chip_lung_support", "chip_lung_score",
              "mutant_minus_control_log2_fpkm1", "welch_p", "bh_q"]
    table = weights.join(evidence[fields], how="left")
    table = table.join(frequency[[f"frequency_{strain}" for strain in STRAINS]], how="left")
    table.index.name = "target"
    return table.reset_index()


def bootstrap_edges(bundle: Path, genes: list[str], regulator: str,
                    alpha: float, max_iter: int, n_bootstrap: int,
                    seed: int) -> pd.DataFrame:
    counts = {strain: np.zeros(len(genes), dtype=int) for strain in STRAINS}
    regulator_index = genes.index(regulator)
    for repetition in range(n_bootstrap):
        data, allowed = _bootstrap_data(bundle, seed + repetition)
        for strain in STRAINS:
            fit = fit_method(data[strain], "ct", alpha, allowed, max_iter)
            counts[strain] += (np.abs(fit.matrix[:, regulator_index]) > 1e-8)
        if (repetition + 1) % 10 == 0:
            print(f"[case] fixed-segmentation bootstrap {repetition + 1}/{n_bootstrap}", flush=True)
    rows = []
    for index, target in enumerate(genes):
        if target == regulator:
            continue
        rows.append({"regulator": regulator, "target": target,
                     **{f"frequency_{strain}": counts[strain][index] / n_bootstrap
                        for strain in STRAINS}})
    return pd.DataFrame(rows)


def draw_figure(table: pd.DataFrame, targets: list[str],
                output: Path, regulator: str) -> None:
    focus = table.set_index("target").loc[targets]
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, (ax_heat, ax_ko) = plt.subplots(
        1, 2, figsize=(9.2, 3.8), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.05, 1.2]},
    )
    matrix = focus[[f"weight_{strain}" for strain in STRAINS]].to_numpy(float)
    vmax = max(0.01, float(np.max(np.abs(matrix))))
    image = ax_heat.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_heat.set_xticks(range(3), STRAINS)
    ax_heat.set_yticks(range(len(targets)), targets)
    ax_heat.set_title("a  Inferred NKX2-1 edges")
    for i in range(len(targets)):
        for j in range(3):
            value = matrix[i, j]
            ax_heat.text(j, i, f"{value:.3f}" if abs(value) > 1e-8 else "0",
                         ha="center", va="center", fontsize=8,
                         color="white" if abs(value) > vmax * 0.53 else "black")
    fig.colorbar(image, ax=ax_heat, shrink=0.72, label="edge coefficient")

    effect = focus.mutant_minus_control_log2_fpkm1.to_numpy(float)
    ax_ko.axvline(0, color="0.4", lw=0.8)
    ax_ko.barh(np.arange(len(targets)), effect, color=["#b35806" if item < 0 else "#5e3c99"
                                                    for item in effect], height=0.62)
    ax_ko.set_yticks(range(len(targets)), targets)
    ax_ko.invert_yaxis()
    ax_ko.set_xlabel("P5 AT1 knockout - control\nlog2(FPKM+1)")
    ax_ko.set_title("b  Independent perturbation response")
    for i, (_, row) in enumerate(focus.iterrows()):
        chip = "ChIP+" if bool(row.chip_lung_support) else "ChIP unrecorded"
        ax_ko.text(0.97, i, chip, va="center", ha="right", fontsize=7.5,
                   transform=ax_ko.get_yaxis_transform(), color="0.2")
    ax_ko.margins(x=0.35)

    fig.suptitle(f"Regulator-centred case analysis: {regulator}", fontsize=12)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Processed real-study directory")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Existing real-data analysis results directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--regulator", default="Nkx2-1")
    parser.add_argument("--targets", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--plot-only", action="store_true",
                        help="Redraw the two-panel case figure from existing focus results")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "nkx21_case_study_ab.pdf"
    if args.plot_only:
        focus = pd.read_csv(args.output_dir / "nkx21_case_focus.tsv", sep="\t")
        draw_figure(focus, focus["target"].tolist(), figure_path, args.regulator)
        print(f"[case] figure: {figure_path}")
        return
    selected = json.loads((args.results_dir / "full_selected_models.json").read_text())
    if selected["discgrn"]["change_time"] is not None:
        raise ValueError("Case-study bootstrap assumes the selected full-data model has one stage")
    alpha = float(selected["discgrn"]["alpha"])
    bundle = args.input_dir / "model/full"
    with np.load(bundle / "trajectories.npz", allow_pickle=False) as archive:
        genes = archive["genes"].astype(str).tolist()
    edges = pd.read_csv(args.results_dir / "full_network_edges.tsv", sep="\t")
    evidence = pd.read_csv(args.results_dir / "nkx21_network_chip_knockout.tsv", sep="\t")
    targets = select_focus(edges, args.regulator, args.targets)
    frequency = bootstrap_edges(bundle, genes, args.regulator, alpha,
                                args.max_iter, args.bootstrap, args.seed)
    frequency.to_csv(args.output_dir / "nkx21_strain_bootstrap_frequency.tsv", sep="\t", index=False)
    table = make_case_table(edges, evidence, args.regulator, frequency)
    table.to_csv(args.output_dir / "nkx21_case_all_targets.tsv", sep="\t", index=False)
    focus = table.set_index("target").loc[targets].reset_index()
    focus.to_csv(args.output_dir / "nkx21_case_focus.tsv", sep="\t", index=False)
    draw_figure(table, targets, figure_path, args.regulator)
    (args.output_dir / "case_configuration.json").write_text(json.dumps({
        "regulator": args.regulator, "target_selection": "top absolute full-data edge coefficients in strongest strain",
        "focus_targets": targets, "selected_alpha": alpha,
        "selected_change_time": None, "bootstrap": args.bootstrap,
        "bootstrap_segmentation": "fixed one-stage model", "max_iter": args.max_iter,
        "seed": args.seed,
    }, indent=2) + "\n")
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
