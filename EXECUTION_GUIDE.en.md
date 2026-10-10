# DiSCGRN manuscript reproduction guide

Run every command from the repository root. All paths in this guide are relative. Parameter sensitivity, change-point evaluation, the five-method directed-edge benchmark, and the lung-development case are distinct experiment batches. Do not pool their system-level records as if they were one sample.

## 1. Environment and external inputs

```bash
conda create -n discgrn python=3.11 -y
conda activate discgrn
python -m pip install -e '.[dev,real,beeline]'
python scripts/setup_data_alias.py
python scripts/check_paper_inputs.py
pytest -q
```

Place the files in the [input inventory](DATA_INVENTORY.en.md) under `Data/`. To extract the specified BEELINE runs from a downloaded archive and verify its MD5, use `python scripts/prepare_beeline_extended_data.py --archive Data/downloads/beeline_datasets.zip --output-root Data/raw/beeline`. Mouse GO, MGI, and KEGG files can be fetched with `python scripts/download_mouse_pathway_annotations.py --output-dir Data/external/mouse_pathway_annotations_20261007`. The GSE74243 and CollecTRI direct links are in the inventory. Database releases change, so exact numerical reproduction should use the recorded release files and checksums.

The Python `beeline` extra installs `arboreto==0.1.6` for GENIE3 and GRNBoost2. Source snapshots for dynGENIE3 and SCODE are under `third_party/`. SCODE and EDISON also need R. In the same Conda environment, run `conda install -c conda-forge r-base r-mass -y`, then `Rscript -e 'install.packages("EDISON", repos="https://cloud.r-project.org")'`, and record `Rscript -e 'sessionInfo()'`; the manuscript batch used EDISON 1.1.2. Full runs are computationally demanding. Where available, use `--limit` or `--max-runs` for a smoke test only, never for the manuscript summaries.

## 2. Parameter-sensitivity panels (paper Figure 1)

```bash
python scripts/run_revision_e3.py --workers 4 --part all
export CTCPGRN_RESULTS_ROOT=results/runs/weighted_score_20261004
python scripts/calibrate_weighted_score.py
unset CTCPGRN_RESULTS_ROOT
python scripts/plot_parameter_sensitivity_panels.py
```

The first command produces sparsity paths and iteration-budget summaries in `results/revision_e3/` from 15 independent systems. The calibration command scans the change penalty on six development systems and writes `rho_grid.tsv` and `selected_parameters.tsv` under `results/runs/weighted_score_20261004/weighted_score_calibration/`. The plotting command writes `sensitivity_alpha.pdf`, `sensitivity_budget.pdf`, and `sensitivity_rho.pdf` to `manuscript/figures/`. `CTCPGRN_RESULTS_ROOT` is a retained internal variable name; its value above is relative.

## 3. Change-point comparison

```bash
export CTCPGRN_RESULTS_ROOT=results/runs/weighted_score_20261004
python scripts/run_revision_comparison.py --workers 4
python scripts/summarize_review_change_points.py
python scripts/run_revision_e4_single.py
python scripts/prepare_revision_e6.py
python scripts/complete_change_point_benchmark.py --part all --workers 4
Rscript scripts/run_revision_e6_external.R frozen_short edison
Rscript scripts/run_revision_e6_external.R e4test edison
python scripts/summarize_change_points_across_lengths.py
unset CTCPGRN_RESULTS_ROOT
```

The first two commands run and summarize the frozen short-sequence comparison. The next three generate the 25- and 41-time-point simulations, matched external inputs, and missing internal fits. The R commands run EDISON on short and long systems. The last command writes the across-length change-point table under `results/runs/weighted_score_20261004/change_points/`. `run_revision_e4_single.py` covers every prespecified change-strength condition and can take substantial time.

## 4. DREAM4-derived directed-edge benchmark (paper Figure 2)

```bash
export CTCPGRN_RESULTS_ROOT=results/runs/weighted_score_20261004
python scripts/prepare_tree_grn_long.py
python scripts/run_tree_grn_benchmark.py --root results/runs/weighted_score_20261004/tree_grn_long
python scripts/run_dyngenie3_long.py --source third_party/dynGENIE3/dynGENIE3.py
unset CTCPGRN_RESULTS_ROOT
python scripts/refit_discgrn_dream_edge_scores.py --workers 4
python scripts/run_grn_accuracy_scode.py --domain dream --rscript "$(command -v Rscript)"
```

The preparation step reconstructs and checks five replicate trajectories for each of 60 fixed-seed systems. The subsequent steps fit GENIE3/GRNBoost2, dynGENIE3, DiSCGRN, and SCODE. DiSCGRN refitting works without an archived historical metric table; if one is present it additionally checks agreement. The tree models can be slow.

## 5. BEELINE and common five-method scoring (paper Figure 3)

The loop processes only the three `dyn-LI` conditions in the paper, with ten runs per condition. Cells are sorted by pseudotime into 13 equal-cell-count bins. The loop fits DiSCGRN and two tree methods; dynGENIE3 and SCODE follow separately.

```bash
for cells in 100 200 500; do
  python scripts/run_beeline_matched_benchmark.py \
    --input-root Data/raw/beeline \
    --pattern "Synthetic/dyn-LI/dyn-LI-${cells}-*" \
    --dataset-label "dyn-LI-${cells}" \
    --output-root "results/beeline_extended_valid_20261009/dyn-LI-${cells}" \
    --methods DiSCGRN GENIE3 GRNBoost2 \
    --alpha 0.0004 --rho 0.30 --budget 80 \
    --tree-python "$(command -v python)"
done
python scripts/run_grn_accuracy_dyngenie3_beeline.py
python scripts/run_grn_accuracy_scode.py --domain beeline --rscript "$(command -v Rscript)"
python scripts/score_five_method_grn_benchmark.py \
  --output-root results/runs/grn_accuracy_20261009/five_method_li_only \
  --beeline-datasets dyn-LI-100 dyn-LI-200 dyn-LI-500
python scripts/plot_five_method_grn_benchmark.py \
  --root results/runs/grn_accuracy_20261009/five_method_li_only
```

Scoring writes `per_run_metrics.tsv`, `summary_by_dataset.tsv`, and `paired_differences.tsv`. It computes AUPR across all directed non-self pairs and Precision, Recall, F1, and F0.1 at the prespecified top 10% of pairs. The paper panels are `dream_five_metrics_dataset_rows.pdf` and `beeline_five_metrics_dataset_rows.pdf` under `five_method_li_only/`. A whole-series baseline score is assessed against each true stage, then averaged within a system.

## 6. GSE74243 mouse lung case (paper Figure 4)

The case derives 500 candidate targets and 30 candidate regulators from GSE74243 and the CollecTRI regulator catalog. Their overlap yields 515 state genes. P0 is a prespecified biological boundary. The real-case settings are `alpha=0.15` and `rho=0.15`; their numerical scale should not be compared directly with the simulation settings.

```bash
python scripts/prepare_three_stage_1000_candidates.py \
  --expression Data/raw/gse74243/GSE74243_matrix_RMAlog2expressionvalues.txt.gz \
  --collectri Data/external/collectri_mouse.tsv \
  --n-targets 500 --n-tfs 30 --pair-mode all_tf_targets \
  --output-dir Data/processed/three_stage_global_500_30tf_allpairs_20261008
python scripts/profile_real_p0_rho.py \
  --candidate-dir Data/processed/three_stage_global_500_30tf_allpairs_20261008 \
  --expression Data/raw/gse74243/GSE74243_matrix_RMAlog2expressionvalues.txt.gz \
  --output-dir results/real_p0_all_tf_targets_20261008 \
  --candidate-mode all_tf_targets --penalty 0.15 --analysis-rho 0.15
python scripts/analyze_two_converged_stage_networks.py \
  --candidate-dir Data/processed/three_stage_global_500_30tf_allpairs_20261008 \
  --prenatal-dir results/real_p0_all_tf_targets_20261008/stage_networks/prenatal \
  --early-dir results/real_p0_all_tf_targets_20261008/stage_networks/early_postnatal \
  --output-dir results/real_p0_all_tf_targets_stage_case_20261008
python scripts/summarize_all_real_regulators.py \
  --candidate-dir Data/processed/three_stage_global_500_30tf_allpairs_20261008 \
  --prenatal-dir results/real_p0_all_tf_targets_20261008/stage_networks/prenatal \
  --early-dir results/real_p0_all_tf_targets_20261008/stage_networks/early_postnatal \
  --output-dir results/real_p0_all_tf_targets_all_regulators_20261008
python scripts/analyze_real_stage_pathways.py \
  --candidate-dir Data/processed/three_stage_global_500_30tf_allpairs_20261008 \
  --edge-dir results/real_p0_all_tf_targets_20261008 \
  --annotation-dir Data/external/mouse_pathway_annotations_20261007 \
  --output-dir results/real_p0_all_tf_targets_pathways_20261008
python scripts/plot_real_allpairs_case_figures.py --composite-panels
```

The profile command fits a whole-period network and two coupled ODE stage networks. The three original fits took roughly two hours in total. Its `*_diagnostics.tsv` files record KKT residuals and convergence. Only stage fits below the `1e-3` tolerance are passed to the biological interpretation scripts. If convergence fails, retain the diagnostics and increase the optimization budget before using the network in a figure. The final command creates six separate PDFs in `results/real_p0_composite_panels_20261008/`: stage UMAP, strain UMAP, regulator bubbles, signed weights, Rb1 GO, and Zbtb20 GO. Complete enrichment tests and gene-set backgrounds remain under `real_p0_all_tf_targets_pathways_20261008/`.

## 7. Figure-only regeneration

If archived **generated outputs** from the [inventory](DATA_INVENTORY.en.md) have been copied to their matching `results/` paths, figures can be regenerated without refitting:

```bash
python scripts/plot_parameter_sensitivity_panels.py
python scripts/plot_five_method_grn_benchmark.py \
  --root results/runs/grn_accuracy_20261009/five_method_li_only
python scripts/plot_real_allpairs_case_figures.py --composite-panels
```

This checks plotting, not an independent numerical rerun. Archive raw files under `Data/` separately from generated `results/`. Date-stamped directory names identify the manuscript batch; they are not run-date requirements.
