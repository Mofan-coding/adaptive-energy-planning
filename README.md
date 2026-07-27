# Adaptive Capacity Expansion under Uncertainty

This repository compares three long-term energy investment strategies under uncertain technology costs:

- **LP Fixed:** one deterministic capacity pathway;
- **LP Myopic:** annual LP decisions that respond to realized costs;
- **DPS Adaptive:** a trained direct-policy-search decision rule.

The repository includes the policy checkpoints and processed CSV/JSON inputs needed to reproduce the main cost-distribution and dominant-technology figures without rerunning the expensive optimization experiments.

## Quick start

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install numpy torch
```

`numpy` and `torch` are listed separately because their pinned entries are intentionally commented out in `requirements.txt`. GLPK is needed only when regenerating annual LP trajectories, not when plotting the included results.

## Reproduce the paper figures

Run all commands from the repository root.

### Dominant-technology frequency bars

```bash
python3 plot_bar.py
```

With the default configuration, this reads the included share and frequency JSON files and writes:

```text
results/figures/122201/dominance_freq_bar_122201_fast transition_2070.png
results/figures/dominance_stacked_cobar_2070_with_lp_det.png
```

### Four-scenario cost distributions

```bash
python3 plot_violin_by_label.py
```

This reads the included LP Fixed and adaptive cost CSVs and writes:

```text
results/figures/combined_cost_violin_4scenarios_dis2.png
```



### Figure 4 technology trajectories

```bash
python3 generation.py
```

This loads the two-SMR policy, runs 100 policy simulations with seed 0, selects the configured learning-rate realization, and writes the Figure 4 panels under:

```text
results/figures/011301/
results/figures/011301/smrs/
```

The first run performs the simulations once in memory; all requested panels reuse those results.

## Paper cases and models

| Case | Label | Transition | EnergySim model |
|---|---|---|---|
| Base | `122201` | Fast | `energySim/energy_sim_model.py` |
| 2 Breakthroughs | `011301` | Fast | `energySim/22_energy_sim_model.py` |
| Risk Averse | `020503` | Fast | `energySim/energy_sim_model.py` |
| Slow Transition | `020502` | Slow | `energySim/energy_sim_model.py` |

The one-SMR model contains `SMR electricity`. The two-SMR model additionally contains `SMR2 electricity`.

## Included reproducibility inputs

Trained policies are stored as:

```text
results/<label>_<scenario>_policy.pth
```

Processed plotting inputs are stored under `results/figures/<label>/`:

```text
<scenario>_cost_violin_data.csv              # LP Fixed costs
<scenario>_1cost_violin_data_dis2.csv        # LP Myopic and DPS costs
shares_exogenous_<label>_<scenario>.json     # LP generation shares
shares_policy_<label>_<scenario>.json        # DPS generation shares
dominance_freq_<label>_<scenario>_<year>.json
```

These files are the minimum inputs for reproducing the aggregate figures. Generated PNG files can be deleted and recreated.


## Repository structure

```text
energySim/                  Energy-system models, parameters, and DPS policy code
LP/                         Annual LP formulations and rollout scripts
results/                    Policy checkpoints and processed figure inputs
plot_bar.py                 Dominant-technology frequency figures
plot_violin_by_label.py     Four-scenario cost-distribution figure
generation.py               Figure 4 trajectory and technology-cost panels
policy_heatmap.py           Policy-response heatmap analysis
```

All scripts use explicit seeds and write outputs beneath `results/figures/`. Configuration is currently file-based rather than command-line based; change only the variables inside each script's `USER CONFIG` block when running a different case.
