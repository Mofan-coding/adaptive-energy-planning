"""Generate Figure 4 for the paper.

Run policy simulations and create the selected deployment, technology-cost,
SMR-share, and technology-legend panels used in Figure 4.
"""

import copy
import importlib
import importlib.util
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

import energySim._energy_sim_params as _energy_sim_params


# ===================== USER CONFIG =====================
nsim = 100
seed = 0
label = "011301"
sim_scenario = "fast transition"

gt_clip = 1.0
hidden_size = 2
input_norm = False

# Figure 4 uses the 12th-highest learning-rate realization by default when
# this file is run as a script. The plotting functions retain rank=10 as their
# public default.
figure_rank = 12
# =======================================================


matplotlib.rc("savefig", dpi=300)
sns.set_style("ticks")
sns.set_context("talk")
matplotlib.rc(
    "font",
    **{"family": "sans-serif", "sans-serif": "Helvetica"},
)


TECH_SOLAR = "solar pv electricity"
TECH_SMR = "SMR electricity"
TECH_SMR2 = "SMR2 electricity"

TECH_COLORS = [
    "black",
    "saddlebrown",
    "darkgray",
    "saddlebrown",
    "darkgray",
    "magenta",
    "royalblue",
    "forestgreen",
    "#7ABBDB",
    "#E29135",
    "#DCA7EB",
    "#984EA3",
    "pink",
    "plum",
    "lawngreen",
    "burlywood",
]


model = None
all_q_policy = []
all_c_policy = []
all_omega_policy = []


def _resolve_model_module(label_value):
    """Return the final EnergySim implementation for the selected label."""
    if label_value == "011301":
        module_name = "energySim.22_energy_sim_model"
    else:
        module_name = "energySim.energy_sim_model"

    try:
        return importlib.import_module(module_name)
    except Exception:
        if module_name != "energySim.22_energy_sim_model":
            raise
        path = os.path.join("energySim", "22_energy_sim_model.py")
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load EnergySim module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _run_policy_simulations():
    """Run and cache the policy realizations required by the Figure 4 plots."""
    global model, all_q_policy, all_c_policy, all_omega_policy

    if all_omega_policy:
        return

    energy_model_module = _resolve_model_module(label)
    model = energy_model_module.EnergyModel(
        EFgp=_energy_sim_params.scenarios[sim_scenario][0],
        slack=_energy_sim_params.scenarios[sim_scenario][1],
        costparams=_energy_sim_params.costsAssumptions["Way et al. (2022)"],
        gt_clip=gt_clip,
        hidden_size=hidden_size,
        input_norm=input_norm,
    )
    model.mode = "policy"

    policy_path = f"results/{label}_{sim_scenario}_policy.pth"
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    model.policy.load(policy_path)

    np.random.seed(seed)
    print(f"Running {nsim} policy simulations for label={label}...")
    for simulation_index in range(nsim):
        model.simulate()
        all_q_policy.append(copy.deepcopy(model.q))
        all_c_policy.append(copy.deepcopy(model.c))
        all_omega_policy.append(copy.deepcopy(model.omega))

        completed = simulation_index + 1
        if completed == 1 or completed % 20 == 0 or completed == nsim:
            print(f"Completed policy simulation {completed}/{nsim}")


def _resolve_technology(tech):
    """Normalize a user-facing technology name to an EnergySim key and tag."""
    tech_input = tech.strip().lower()
    if tech_input in {"solar", "solar pv electricity", "solar electricity"}:
        return TECH_SOLAR, "Solar"
    if tech_input in {
        "smr",
        "smr electricity",
        "small modular reactor",
        "small modular reactor electricity",
    }:
        return TECH_SMR, "SMR"
    if tech_input in {
        "smr2",
        "smr2 electricity",
        "small modular reactor2",
        "small modular reactor 2",
    }:
        return TECH_SMR2, "SMR2"
    raise ValueError(f'Unknown technology: {tech}. Use "solar", "SMR", or "SMR2".')


def _ranked_policy_sample(tech, rank):
    """Return the sample index for the requested descending learning-rate rank."""
    if rank < 1:
        raise ValueError("rank must be at least 1")

    _run_policy_simulations()
    technology_key, technology_tag = _resolve_technology(tech)
    learning_rates = np.array(
        [omega.get(technology_key, np.nan) for omega in all_omega_policy],
        dtype=float,
    )
    valid_indices = np.flatnonzero(~np.isnan(learning_rates))
    if len(valid_indices) < rank:
        raise ValueError(
            f"Not enough valid policy samples for {technology_tag} rank={rank}; "
            f"only {len(valid_indices)} are available."
        )

    descending = valid_indices[np.argsort(learning_rates[valid_indices])[::-1]]
    sample_index = int(descending[rank - 1])
    return sample_index, learning_rates, technology_tag


def _years():
    return list(range(model.y0, model.yend + 1))


def _series(data, technology, length, fill_value):
    """Return a one-dimensional technology series with a predictable length."""
    values = np.asarray(data.get(technology, [fill_value] * length), dtype=float)
    if len(values) < length:
        values = np.pad(
            values,
            (0, length - len(values)),
            constant_values=fill_value,
        )
    return values[:length].copy()


def _cost_series(cost_data, technology, years):
    values = _series(cost_data, technology, len(years), np.nan)
    if technology in {TECH_SMR, TECH_SMR2}:
        values[np.asarray(years) < 2030] = np.nan
    return values


def plot_policy_tj_cost_top10(tech="SMR electricity", rank=10):
    """Plot deployment and cost paths for a ranked policy realization.

    The realization is selected using the requested technology's learning
    rate. Three files are created: final energy by source, Solar/SMR/SMR2 unit
    costs, and Solar/SMR generation trajectories.
    """
    sample_index, learning_rates, technology_tag = _ranked_policy_sample(tech, rank)
    years = _years()
    save_dir = f"results/figures/{label}"
    os.makedirs(save_dir, exist_ok=True)

    generation_data = all_q_policy[sample_index]
    generation_df = pd.DataFrame(generation_data, index=years)
    excluded = {"qgrid", "qtransport", "electricity networks", "electrolyzers"}
    generation_df = generation_df[
        [column for column in generation_df.columns if column not in excluded]
    ]

    fig_energy, ax_energy = plt.subplots(figsize=(12, 6))
    generation_df.plot.area(
        stacked=True,
        linewidth=0,
        ax=ax_energy,
        color=TECH_COLORS[: len(generation_df.columns)],
        legend=False,
    )
    ax_energy.set_xlim(2020, 2070)
    ax_energy.set_ylim(0, 1500)
    ax_energy.set_ylabel("Generation (EJ)", fontsize=28, weight="bold")
    ax_energy.set_xlabel("Year", fontsize=28, weight="bold")
    ax_energy.tick_params(axis="both", labelsize=24)
    fig_energy.tight_layout()
    fig_energy.savefig(
        f"{save_dir}/policy_top{rank}_{technology_tag.lower()}_final_energy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_energy)

    cost_data = all_c_policy[sample_index]
    solar_cost = _cost_series(cost_data, TECH_SOLAR, years)
    smr_cost = _cost_series(cost_data, TECH_SMR, years)
    smr2_cost = _cost_series(cost_data, TECH_SMR2, years)

    fig_cost, ax_cost = plt.subplots(figsize=(12, 6))
    ax_cost.plot(years, solar_cost, label="Solar PV", color="#E29135", linewidth=6)
    ax_cost.plot(years, smr_cost, label="SMR (≥2030)", color="#DCA7EB", linewidth=6)
    ax_cost.plot(years, smr2_cost, label="SMR2 (≥2030)", color="#984EA3", linewidth=6)
    ax_cost.set_xlabel("Year", fontsize=28, weight="bold")
    ax_cost.set_ylabel("Unit Cost (USD/GJ)", fontsize=28, weight="bold")
    ax_cost.set_xlim(2020, 2070)
    ax_cost.tick_params(axis="both", labelsize=24)
    fig_cost.tight_layout()
    fig_cost.savefig(
        f"{save_dir}/policy_top{rank}_{technology_tag.lower()}_cost_evolution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_cost)

    solar_generation = _series(generation_data, TECH_SOLAR, len(years), 0.0)
    smr_generation = _series(generation_data, TECH_SMR, len(years), 0.0)

    fig_generation, ax_generation = plt.subplots(figsize=(12, 6))
    ax_generation.plot(
        years,
        solar_generation,
        label="Solar generation",
        color="tab:orange",
        linewidth=4,
    )
    ax_generation.plot(
        years,
        smr_generation,
        label="SMR generation",
        color="tab:blue",
        linewidth=4,
    )
    ax_generation.set_title(
        "Solar vs. SMR Generation over Time",
        fontsize=28,
        weight="bold",
    )
    ax_generation.set_xlabel("Year", fontsize=24, weight="bold")
    ax_generation.set_ylabel("Generation (EJ)", fontsize=24, weight="bold")
    ax_generation.set_xlim(2020, 2070)
    ax_generation.tick_params(axis="both", labelsize=22)
    ax_generation.legend(loc="best", fontsize=22, frameon=False)
    fig_generation.tight_layout()
    fig_generation.savefig(
        f"{save_dir}/policy_top{rank}_{technology_tag.lower()}_solar_smr_qtimeseries.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_generation)

    print(
        f"{technology_tag} {rank}th-highest learning-rate sample: "
        f"index={sample_index}, rate={learning_rates[sample_index]:.6f}"
    )


def plot_tech_legend_only(label="122201"):
    """Create the standalone technology-color legend used in Figure 4."""
    technologies = [
        "oil (direct use)",
        "coal (direct use)",
        "gas (direct use)",
        "coal electricity",
        "gas electricity",
        "nuclear electricity",
        "hydroelectricity",
        "biopower electricity",
        "wind electricity",
        "solar pv electricity",
        "SMR electricity",
        "SMR2 electricity",
        "daily batteries",
        "multi-day storage",
        "electrolyzers",
    ]
    colors = TECH_COLORS[: len(technologies)]

    handles = [Line2D([0], [0], color=color, linewidth=8) for color in colors]
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")
    ax.legend(
        handles,
        technologies,
        loc="center",
        ncol=5,
        fontsize=18,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.6,
    )
    fig.tight_layout()

    save_dir = f"results/figures/{label}"
    os.makedirs(save_dir, exist_ok=True)
    output_path = f"{save_dir}/tech_color_legend.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Technology legend saved to {output_path}")


def plot_smrs_cost_top10(tech="SMR electricity", rank=10):
    """Plot ranked Solar/SMR/SMR2 costs with SMR generation-share areas."""
    sample_index, learning_rates, technology_tag = _ranked_policy_sample(tech, rank)
    years = _years()
    cost_data = all_c_policy[sample_index]
    generation_data = all_q_policy[sample_index]

    solar_cost = _cost_series(cost_data, TECH_SOLAR, years)
    smr_cost = _cost_series(cost_data, TECH_SMR, years)
    smr2_cost = _cost_series(cost_data, TECH_SMR2, years)

    fig, ax_cost = plt.subplots(figsize=(12, 6))
    ax_cost.plot(years, solar_cost, label="Solar PV", color="#E29135", linewidth=6)
    ax_cost.plot(years, smr_cost, label="SMR (≥2030)", color="#DCA7EB", linewidth=6)
    ax_cost.plot(years, smr2_cost, label="SMR2 (≥2030)", color="#984EA3", linewidth=6)
    ax_cost.set_xlabel("Year", fontsize=28, weight="bold")
    ax_cost.set_ylabel("Unit Cost (USD/GJ)", fontsize=28, weight="bold")
    ax_cost.set_xlim(2020, 2070)
    ax_cost.tick_params(axis="both", labelsize=24)

    smr_generation = _series(generation_data, TECH_SMR, len(years), 0.0)
    smr2_generation = _series(generation_data, TECH_SMR2, len(years), 0.0)
    total_generation = smr_generation + smr2_generation
    with np.errstate(divide="ignore", invalid="ignore"):
        smr_share = np.where(total_generation > 0, smr_generation / total_generation, np.nan)
        smr2_share = np.where(total_generation > 0, smr2_generation / total_generation, np.nan)

    share_mask = np.asarray(years) >= 2030
    ax_share = ax_cost.twinx()
    ax_share.set_ylim(0, 1.0)
    ax_share.set_ylabel("SMR vs. SMR2 Share", fontsize=28, weight="bold")
    ax_share.tick_params(axis="y", labelsize=24)
    ax_share.stackplot(
        np.asarray(years)[share_mask],
        smr_share[share_mask],
        smr2_share[share_mask],
        labels=["SMR share", "SMR2 share"],
        colors=["#E0B2ED", "#71207D"],
        alpha=0.2,
    )

    handles_cost, labels_cost = ax_cost.get_legend_handles_labels()
    handles_share, labels_share = ax_share.get_legend_handles_labels()
    unique_entries = {}
    for handle, legend_label in zip(
        handles_cost + handles_share,
        labels_cost + labels_share,
    ):
        unique_entries.setdefault(legend_label, handle)
    ax_cost.legend(
        unique_entries.values(),
        unique_entries.keys(),
        loc="best",
        fontsize=23,
        frameon=False,
    )

    fig.tight_layout()
    save_dir = f"results/figures/{label}/smrs"
    os.makedirs(save_dir, exist_ok=True)
    output_path = (
        f"{save_dir}/smrs_cost_share_top{rank}_{technology_tag.lower()}_idx{sample_index}.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        f"SMR cost/share plot saved to {output_path}; "
        f"{technology_tag} rank={rank}, index={sample_index}, "
        f"rate={learning_rates[sample_index]:.6f}"
    )


if __name__ == "__main__":
    plot_policy_tj_cost_top10("SMR electricity", rank=figure_rank)
    plot_tech_legend_only(label=label)
    plot_smrs_cost_top10("SMR electricity", rank=figure_rank)
