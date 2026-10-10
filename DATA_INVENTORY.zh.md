# 论文输入数据清单

以下均为**项目内相对路径**。把当前工作区相应文件复制到新项目的 `Data/` 下，保留子目录。运行 `python scripts/check_paper_inputs.py` 可检查正文必须的外部文件。生成的 `Data/processed/` 与 `results/` 不属于原始下载数据。

| 用途 | 复制到项目内 | 原始下载地址 |
|---|---|---|
| DREAM4 主拓扑 | `Data/raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv` | [GeneNetWeaver DREAM4 压缩包](https://gnw.sourceforge.net/resources/DREAM4%20in%20silico%20challenge.zip)；[说明页](https://gnw.sourceforge.net/dreamchallenge.html) |
| BEELINE 30 个实例 | `Data/raw/beeline/Synthetic/dyn-LI/dyn-LI-{100,200,500}-{1..10}/`，每个实例含 `ExpressionData.csv`、`PseudoTime.csv`、`GroundTruth.csv`、`refNetwork.csv` | [Zenodo 7682713 的 `beeline_datasets.zip`](https://zenodo.org/records/7682713)；压缩包 MD5 `857cb335da79d09865cd7db2549fbda4` |
| 小鼠肺发育表达 | `Data/raw/gse74243/GSE74243_matrix_RMAlog2expressionvalues.txt.gz` | [GEO 补充文件](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE74nnn/GSE74243/suppl/GSE74243_matrix_RMAlog2expressionvalues.txt.gz) |
| GSE74243 样本与探针资料（审计用） | `Data/raw/gse74243/GSE74243_series_matrix.txt.gz`、`GSE74243_Probes_BultLung2015.tsv.gz` | [GEO 系列矩阵](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE74nnn/GSE74243/matrix/GSE74243_series_matrix.txt.gz)、[探针文件](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE74nnn/GSE74243/suppl/GSE74243_Probes_BultLung2015.tsv.gz) |
| 可测候选调控因子 | `Data/external/collectri_mouse.tsv` | [OmniPath 小鼠 CollecTRI API](https://omnipathdb.org/interactions?datasets=tf_target&resources=CollecTRI&organisms=10090&format=tsv&fields=sources,references&genesymbols=1) |
| GO 小鼠注释与本体 | `Data/external/mouse_pathway_annotations_20261007/MOUSE-mod.gaf.gz`、`go-basic.obo` | [GO 小鼠 GAF](https://current.geneontology.org/annotations/gaf/MOUSE-mod.gaf.gz)、[GO 本体](https://current.geneontology.org/ontology/go-basic.obo) |
| MGI ID 映射 | `Data/external/mouse_pathway_annotations_20261007/MGI_Gene_Model_Coord.rpt` | [MGI 报告](https://www.informatics.jax.org/downloads/reports/MGI_Gene_Model_Coord.rpt) |
| KEGG 小鼠通路与基因 | `Data/external/mouse_pathway_annotations_20261007/kegg_mouse_pathways.tsv`、`kegg_mouse_pathway_genes.tsv` | [通路表](https://rest.kegg.jp/list/pathway/mmu)、[通路—基因连接](https://rest.kegg.jp/link/mmu/pathway) |
| 注释版本记录 | `Data/external/mouse_pathway_annotations_20261007/manifest.json` | 由 `scripts/download_mouse_pathway_annotations.py` 生成，含下载时校验值 |

为了检查历史扩展分析，可一并复制整个 `Data/raw/dream4/`（其余四张 10 基因拓扑和原生时间序列）；它们并非正文主模拟轨迹的表达输入。`Data/processed/three_stage_global_500_30tf_allpairs_20261008/` 是 GSE74243 与 CollecTRI 经基因筛选后生成的 515 基因面板，可重新生成，不要求另行下载。

若**只想复刻已发表批次的图而不重算耗时模型**，还需把原项目中的以下生成结果复制到新项目同名 `results/` 相对路径：`revision_e3/`、`runs/weighted_score_20261004/`、`beeline_extended_valid_20261009/`、`runs/grn_accuracy_20261009/`、`real_p0_all_tf_targets_20261008/`、`real_p0_all_tf_targets_stage_case_20261008/`、`real_p0_all_tf_targets_all_regulators_20261008/`、`real_p0_all_tf_targets_pathways_20261008/`。这些是**可再生成的输出缓存**，不要误称为外部原始数据。部分脚本拒绝覆盖既有输出；从头运行时应保持 `results/` 为空。
