# DiSCGRN：连续时间基因调控网络的离散切换推断

DiSCGRN（Discrete Switching in Continuous-time Gene Regulatory Network Inference）使用分段仿射常微分方程和实际采样间隔的矩阵指数传播估计阶段网络，并比较无变化与一次变化模型。本仓库整理了论文当前使用的算法、模拟、对照、评价和绘图代码。内部 Python 包名仍为 `ctcpgrn`，以保持已完成实验的导入和运行记录不变。

论文的三部分证据包括：DREAM4 拓扑派生的非平稳模拟及变化点实验，BEELINE `dyn-LI-100/200/500` 伪时间基准，以及 GSE74243 小鼠肺发育的固定 P0 阶段案例。BEELINE 的伪时间分箱并不等于同一个体的纵向时间序列。真实案例的候选边是表达动力学中的有效预测关系，不能直接视为已证实的物理结合。

## 目录

| 路径 | 内容 |
|---|---|
| `src/ctcpgrn/` | 模型、仿真、指标和区间搜索 |
| `scripts/` | 数据准备、基准、真实案例、作图与输入检查 |
| `third_party/` | 本研究实际调用的 dynGENIE3 与 SCODE 源码快照 |
| `tests/` | 单元与回归测试 |
| `Data/` | 用户复制或下载的外部输入，以及随后生成的处理数据 |
| `results/` | 运行输出；初始为空 |
| `manuscript/figures/` | 论文参数敏感度图的默认输出位置 |

项目采用大写 `Data/` 存放数据。旧实验脚本使用小写 `data/`；先运行 `python scripts/setup_data_alias.py` 即可兼容区分大小写的文件系统。本仓库不随代码再分发外部数据，见 [数据清单](DATA_INVENTORY.zh.md)。

## 开始使用

```bash
conda create -n discgrn python=3.11 -y
conda activate discgrn
python -m pip install -e '.[dev,real,beeline]'
python scripts/setup_data_alias.py
python scripts/check_paper_inputs.py
pytest -q
```

不下载数据也可先运行 `python scripts/run_smoke_test.py`，验证模拟、拟合及指标计算链路。`check_paper_inputs.py` 在 `Data/` 尚为空时退出并列出缺失文件，这是预期行为。

获得或复制数据后，按 [完整执行手册](EXECUTION_GUIDE.zh.md) 运行。正式图表采用论文指定的原始种子、参数与固定输入，不应以测试真值重新调参。各第三方数据及软件应遵守其来源许可并在论文中分别引用。
