# DiSCGRN 论文复现执行手册

所有命令均从仓库根目录执行，使用相对路径。下面区分“从原始输入重新计算”和“已有输出时只重绘”。论文正文的参数敏感度、变化点比较、五方法有向边基准和肺发育案例由不同批次构成，请勿把不同批次的系统级结果拼为一个样本量。

## 1. 环境与输入

```bash
conda create -n discgrn python=3.11 -y
conda activate discgrn
python -m pip install -e '.[dev,real,beeline]'
python scripts/setup_data_alias.py
python scripts/check_paper_inputs.py
pytest -q
```

将 [数据清单](DATA_INVENTORY.zh.md) 中的原始文件放到 `Data/`。若自行下载 BEELINE 完整压缩包，可用 `python scripts/prepare_beeline_extended_data.py --archive Data/downloads/beeline_datasets.zip --output-root Data/raw/beeline` 提取指定实例并校验 MD5。小鼠 GO、MGI 和 KEGG 注释可用 `python scripts/download_mouse_pathway_annotations.py --output-dir Data/external/mouse_pathway_annotations_20261007` 下载。GSE74243 与 CollecTRI 可依清单直链下载，或查看 `python scripts/download_real_data.py --help`。外部数据库会更新，严格数值复现应优先使用原批次文件及其校验清单。

树模型需要 `arboreto==0.1.6`（上面的 `beeline` 可选依赖）；dynGENIE3 源码和 SCODE 源码已在 `third_party/`。SCODE 与 EDISON 还需要 R。在同一 Conda 环境中运行 `conda install -c conda-forge r-base r-mass -y`，再运行 `Rscript -e 'install.packages("EDISON", repos="https://cloud.r-project.org")'` 并保存 `Rscript -e 'sessionInfo()'` 的输出；EDISON 论文批次使用 1.1.2。计算量较大，建议先以各运行器的 `--limit` 或 `--max-runs` 做冒烟检查，不要把限量输出用于论文汇总。

## 2. 参数敏感度图（正文图 1）

```bash
python scripts/run_revision_e3.py --workers 4 --part all
export CTCPGRN_RESULTS_ROOT=results/runs/weighted_score_20261004
python scripts/calibrate_weighted_score.py
unset CTCPGRN_RESULTS_ROOT
python scripts/plot_parameter_sensitivity_panels.py
```

`run_revision_e3.py` 生成 15 个独立系统上的稀疏度路径与迭代预算表，输出 `results/revision_e3/`。`calibrate_weighted_score.py` 在六个开发系统上扫描变化惩罚，输出 `results/runs/weighted_score_20261004/weighted_score_calibration/` 的 `rho_grid.tsv` 和 `selected_parameters.tsv`。最后一条命令读取两批表，生成 `manuscript/figures/sensitivity_alpha.pdf`、`sensitivity_budget.pdf` 和 `sensitivity_rho.pdf`。这里的 `CTCPGRN_RESULTS_ROOT` 是旧代码保留的环境变量名，其值仍是相对路径。

## 3. 变化点比较

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

首先在独立短序列上运行固定参数的 DiSCGRN 及内部对照，输出 `revision_comparison_h80/` 和短序列统计。接着生成长度 25 与 41 的变化强度实验、外部方法共用输入，并补齐所需内部方法。两条 R 命令分别在短序列和长序列系统上运行 EDISON。最后生成 `results/runs/weighted_score_20261004/change_points/` 下的跨长度比较表。`run_revision_e4_single.py` 扫描全部预设变化强度条件，因此显著耗时；若只复制历史结果，直接使用第 6 节的重绘路径。

## 4. DREAM4 派生的五方法有向边基准（正文图 2）

```bash
export CTCPGRN_RESULTS_ROOT=results/runs/weighted_score_20261004
python scripts/prepare_tree_grn_long.py
python scripts/run_tree_grn_benchmark.py --root results/runs/weighted_score_20261004/tree_grn_long
python scripts/run_dyngenie3_long.py --source third_party/dynGENIE3/dynGENIE3.py
unset CTCPGRN_RESULTS_ROOT
python scripts/refit_discgrn_dream_edge_scores.py --workers 4
python scripts/run_grn_accuracy_scode.py --domain dream --rscript "$(command -v Rscript)"
```

`prepare_tree_grn_long.py` 从相同的 DREAM4 拓扑和固定种子恢复 60 个系统的五条轨迹，并核对上一步外部输入。随后分别拟合 GENIE3/GRNBoost2、dynGENIE3、DiSCGRN 和 SCODE。SCODE 采用固定种子重复并输出有向得分。新项目的 DiSCGRN 重拟合不要求旧论文统计表存在；若旧表存在，会额外检查分数一致性。树模型运行较慢。

## 5. BEELINE 基准与五方法统一评分（正文图 3）

下面的循环只运行正文保留的三个 `dyn-LI` 条件，每个条件含 10 个实例，并将单细胞按伪时间分为 13 个等细胞数箱。运行器只计算正文需要的 DiSCGRN、GENIE3 和 GRNBoost2；另两种方法在后续单独运行。

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

统一评分生成 `per_run_metrics.tsv`、`summary_by_dataset.tsv`、`paired_differences.tsv`，指标为 AUPR 和预设前 10% 有向候选边的 Precision、Recall、F1、F0.1。两张论文图分别是 `dream_five_metrics_dataset_rows.pdf` 与 `beeline_five_metrics_dataset_rows.pdf`，都位于 `five_method_li_only/`。静态方法的一张得分网络分别与两个真实阶段比较，再在系统内取均值。

## 6. GSE74243 肺发育案例（正文图 4）

案例从 GSE74243 表达和 CollecTRI 调控因子目录产生 500 个候选靶基因与 30 个调控因子；两组重叠后形成 515 个状态基因。P0 是预设的生物学边界。真实案例采用 `alpha=0.15`、`rho=0.15`，与模拟参数不能直接比较。

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

第二条命令拟合全程、胚胎期、出生后早期的耦合连续时间网络。原批次三次拟合总计约耗时两小时；`results/real_p0_all_tf_targets_20261008/*_diagnostics.tsv` 记录 KKT 残差和收敛状态，仅通过阈值 `1e-3` 的阶段拟合可进入后续解释。若未收敛，应保存诊断并增加优化预算，不得把未收敛网络直接用于图表。最后生成六个独立 PDF，位于 `results/real_p0_composite_panels_20261008/`，依次为阶段 UMAP、品系 UMAP、调控因子气泡图、边权热图、Rb1 GO 和 Zbtb20 GO。富集完整检验与背景集合保存在 `real_p0_all_tf_targets_pathways_20261008/`。

## 7. 只重绘与核对

若已按 [数据清单](DATA_INVENTORY.zh.md) 将旧批次**生成结果**复制到同名 `results/` 目录，则无需重跑模型即可执行：

```bash
python scripts/plot_parameter_sensitivity_panels.py
python scripts/plot_five_method_grn_benchmark.py \
  --root results/runs/grn_accuracy_20261009/five_method_li_only
python scripts/plot_real_allpairs_case_figures.py --composite-panels
```

这只验证作图链，不是独立的从头数值复现。`Data/` 中原始文件与 `results/` 中生成结果应分别归档。所有固定日期的目录名是论文批次标识，不是脚本必须运行于该日期的要求。
