#!/usr/bin/env python3
"""Plot fixed-P0 rho sensitivity from a converged whole fit and feasible split.

The two previously converged stage fits are valid parameter choices in the
shared union candidate universe, but are not re-optimised there. Their summed
cost is therefore an upper bound on the attainable two-stage fitted cost.
This gives a conservative sensitivity threshold for the *fitted* whole model.
It is not a search over unknown boundaries or a global-optimum certificate.
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


def read_one(path: Path, stage: str, n: int) -> pd.Series:
    frame = pd.read_csv(path, sep="\t")
    row = frame.loc[frame.stage.eq(stage)]
    if len(row) != 1 or not bool(row.iloc[0].converged):
        raise ValueError(f"{stage} has no converged diagnostic in {path}")
    if int(row.iloc[0].n_transitions) != n:
        raise ValueError(f"Expected {n} transitions for {stage}")
    return row.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path,
                        default=Path("results/real_p0_rho_profile_common_candidates_20261007"))
    parser.add_argument("--whole-diagnostics", type=Path)
    parser.add_argument("--prenatal-dir", type=Path,
                        default=Path("results/global_three_stage_500_30tf_prenatal_active_strict_20261007"))
    parser.add_argument("--early-dir", type=Path,
                        default=Path("results/global_three_stage_500_30tf_early_active_strict_20261007"))
    args = parser.parse_args()
    whole_path = (args.whole_diagnostics or args.profile_dir / "whole_diagnostics.tsv")
    whole = read_one(whole_path, "whole", 45)
    prenatal = read_one(args.prenatal_dir / "fit_diagnostics.tsv", "prenatal", 24)
    early = read_one(args.early_dir / "fit_diagnostics.tsv", "early_postnatal", 21)
    configs = [json.loads((directory / "run_configuration.json").read_text())
               for directory in (args.prenatal_dir, args.early_dir)]
    profile = json.loads((args.profile_dir / "profile_configuration.json").read_text())
    if any(abs(float(config["penalty"]) - float(profile["alpha"])) > 1e-12
           for config in configs):
        raise ValueError("Stage and whole fits use different alpha values")
    l0 = float(whole.additive_cost)
    split_feasible = (24 * float(prenatal.objective)
                      + 21 * float(early.objective))
    if l0 <= 0:
        raise ValueError("The no-change cost must be positive")
    rho_crit = (l0 - split_feasible) / l0
    rho = np.linspace(0, 0.9, 181)
    frame = pd.DataFrame({
        "rho": rho,
        "whole_fitted_cost": l0,
        "p0_feasible_split_penalised_cost": split_feasible + rho * l0,
    })
    frame["existing_p0_fit_better_than_fitted_whole"] = (
        frame.p0_feasible_split_penalised_cost < frame.whole_fitted_cost)
    frame.to_csv(args.profile_dir / "rho_profile_feasible.tsv", sep="\t", index=False)
    summary = {
        "whole_fitted_cost_L0": l0,
        "p0_existing_stage_feasible_cost": split_feasible,
        "rho_conservative_threshold": rho_crit,
        "selected_for": "Fixed P0, two-stage feasible fit versus converged whole fit",
        "important_limits": [
            "The two stages were fit in their original stage-specific candidate masks and have not been re-optimised in the union mask",
            "This is a sufficient fitted-cost threshold, not an exact optimum over the union mask",
            "The P0 boundary was chosen in advance; alternative boundaries were not scanned",
            "The affine ODE objective is nonconvex, so a small KKT residual does not certify a global optimum",
            "A post hoc rho threshold is not an independently validated change-point discovery",
        ],
    }
    (args.profile_dir / "rho_profile_feasible_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(rho, frame.whole_fitted_cost, color="#355C7D", lw=2,
            label="Converged fitted no-change cost")
    ax.plot(rho, frame.p0_feasible_split_penalised_cost, color="#D97A43", lw=2,
            label="Existing P0 split cost + rho penalty")
    if 0 <= rho_crit <= 0.9:
        ax.axvline(rho_crit, color="#4C4C4C", ls="--", lw=1.1,
                   label=f"Conservative threshold = {rho_crit:.3f}")
    ax.set(xlabel=r"Change-point penalty ratio $\rho$",
           ylabel="Additive fitted cost",
           title="Fixed-P0 rho sensitivity (feasible split)")
    ax.legend(frameon=False)
    fig.savefig(args.profile_dir / "rho_profile_feasible.pdf")
    fig.savefig(args.profile_dir / "rho_profile_feasible.png", dpi=220)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
