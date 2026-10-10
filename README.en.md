# DiSCGRN: discrete switching in continuous-time GRN inference

DiSCGRN fits stage-specific affine ODE networks using matrix-exponential propagation over the observed sampling intervals. It then compares a no-change model with a one-change model. This repository collects the algorithm, simulations, baselines, evaluation, and plotting code used in the current manuscript. The internal Python package remains named `ctcpgrn` to preserve the recorded experiment imports.

The manuscript uses three evidence streams: nonstationary simulations derived from DREAM4 topology, the BEELINE `dyn-LI-100/200/500` pseudotime benchmark, and a fixed-P0 mouse lung development case from GSE74243. BEELINE pseudotime bins are not longitudinal observations of the same cell. Edges inferred from the real expression data are effective dynamical associations, not confirmed physical binding events.

## Layout

| Path | Purpose |
|---|---|
| `src/ctcpgrn/` | Model, simulation, metrics, and interval search |
| `scripts/` | Input preparation, benchmarks, case study, figures, and checks |
| `third_party/` | Source snapshots of dynGENIE3 and SCODE used here |
| `tests/` | Unit and regression tests |
| `Data/` | User-supplied external inputs and generated processed data |
| `results/` | Generated outputs; initially empty |
| `manuscript/figures/` | Default location for parameter-sensitivity panels |

External datasets are not redistributed with the code. See the [input inventory](DATA_INVENTORY.en.md). Historical scripts refer to lowercase `data/`; run `python scripts/setup_data_alias.py` once to make those paths resolve to `Data/` on case-sensitive systems.

## Quick start

```bash
conda create -n discgrn python=3.11 -y
conda activate discgrn
python -m pip install -e '.[dev,real,beeline]'
python scripts/setup_data_alias.py
python scripts/check_paper_inputs.py
pytest -q
```

Before downloading data, `python scripts/run_smoke_test.py` checks simulation, fitting, and scoring with a small in-memory system. `check_paper_inputs.py` is expected to exit with a missing-file list while `Data/` is empty.

After acquiring the inputs, follow the [full execution guide](EXECUTION_GUIDE.en.md). Reproduction uses the recorded seeds and parameters. Do not tune on held-out ground truth. Follow the licensing and citation requirements of each upstream dataset and method.
