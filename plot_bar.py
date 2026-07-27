"""Generate figure 3 for the paper.

The script can simulate and save LP/DPS generation shares for one case, build
its dominance-frequency JSON, plot the case-level comparison, and combine the
saved results from four cases into a paper-ready stacked bar figure.
"""

import importlib
import importlib.util
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import energySim._energy_sim_params as _energy_sim_params


# ===================== USER CONFIG =====================
simulate = False
plot_bar = True
plot_combined_bar = True
include_lp_fixed = True

label = "122201"
target_year = 2070
summary_years = (2050, 2070)
nsim = 500
seed = 0

use_lp_folder = True
lp_folder = ""

gt_clip = 1.0
hidden_size = 2
input_norm = False
# =======================================================


TECHNOLOGIES = [
    "solar pv electricity",
    "wind electricity",
    "hydroelectricity",
    "SMR electricity",
    "SMR2 electricity",
]

STATE_LABELS = {
    "solar pv electricity": "Solar",
    "wind electricity": "Wind",
    "hydroelectricity": "Hydro",
    "SMR electricity": "SMR",
    "SMR2 electricity": "SMR2",
    "mix": "Mix",
}

COLORS = {
    "solar pv electricity": "#E29135",
    "wind electricity": "#7ABBDB",
    "hydroelectricity": "royalblue",
    "SMR electricity": "#DCA7EB",
    "SMR2 electricity": "#984EA3",
    "mix": "#A5AEB7",
}

COMPARISON_LABELS = ["122201", "011301", "020503", "020502"]
COMPARISON_NAMES = ["Base", "2 Breakthroughs", "Risk Averse", "Slow Transition"]


def resolve_label_config(label_value):
    """Return the final EnergySim module and scenario for a label."""
    if label_value in {"011301", "011903", "011903-s"}:
        return "energySim.22_energy_sim_model", "fast transition"
    if label_value == "020502":
        return "energySim.energy_sim_model", "slow transition"
    return "energySim.energy_sim_model", "fast transition"


def load_energy_model_module(module_name):
    """Load an EnergySim module, including modules with numeric filenames."""
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


def get_lp_folder_for_label(label_value):
    """Return the per-simulation LP trajectory folder for a label."""
    folders = {
        "011301": "results/_11_lp_nsim_runs",
        "020503": "results/_22_lp_nsim_runs",
        "020502": "results/_33_lp_nsim_runs",
    }
    return folders.get(label_value, "results/_00_lp_nsim_runs")


def get_lp_files_for_runs(label_value, count):
    """Return and validate the LP trajectory files needed for simulation."""
    if not use_lp_folder:
        raise ValueError("plot_bar.py requires use_lp_folder=True")

    folder = lp_folder or get_lp_folder_for_label(label_value)
    files = []
    for index in range(1, count + 1):
        path = os.path.join(folder, f"sim_{index:04d}_results_lp_trajectory.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing LP simulation file: {path}")
        files.append(path)
    return files


def get_dominance_state(shares, technologies=TECHNOLOGIES):
    """Return the technology above 50% share, or ``mix`` if none qualifies."""
    values = [float(shares.get(technology, 0.0)) for technology in technologies]
    if not values:
        return "mix"
    maximum = max(values)
    if maximum > 0.5:
        return technologies[int(np.argmax(values))]
    return "mix"


def run_policy_shares(module_name, scenario_name):
    """Simulate DPS and return generation-share trajectories for all runs."""
    energy_model_module = load_energy_model_module(module_name)
    model = energy_model_module.EnergyModel(
        EFgp=_energy_sim_params.scenarios[scenario_name][0],
        slack=_energy_sim_params.scenarios[scenario_name][1],
        costparams=_energy_sim_params.costsAssumptions["Way et al. (2022)"],
        gt_clip=gt_clip,
        hidden_size=hidden_size,
        input_norm=input_norm,
    )
    model.mode = "policy"

    policy_path = f"results/{label}_{scenario_name}_policy.pth"
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    model.policy.load(policy_path)

    np.random.seed(seed)
    results = []
    for index in range(nsim):
        model.simulate()
        results.append(model.get_generation_shares().to_dict(orient="index"))
        completed = index + 1
        if completed == 1 or completed % 20 == 0 or completed == nsim:
            print(f"Completed DPS share simulation {completed}/{nsim}")
    return results


def run_exogenous_shares(module_name, scenario_name, lp_files):
    """Simulate the per-run LP trajectories and return their generation shares."""
    energy_model_module = load_energy_model_module(module_name)
    model = energy_model_module.EnergyModel(
        EFgp=_energy_sim_params.scenarios[scenario_name][0],
        slack=_energy_sim_params.scenarios[scenario_name][1],
        costparams=_energy_sim_params.costsAssumptions["Way et al. (2022)"],
        gt_clip=gt_clip,
        hidden_size=hidden_size,
        input_norm=input_norm,
    )
    model.mode = "exogenous"

    results = []
    for index, lp_file in enumerate(lp_files, start=1):
        model.lp_trajectory_file = lp_file
        model.load_lp_trajectories_from_file(lp_file)
        model.simulate()
        results.append(model.get_generation_shares().to_dict(orient="index"))
        if index == 1 or index % 20 == 0 or index == len(lp_files):
            print(f"Completed LP share simulation {index}/{len(lp_files)}")
    return results


def _get_year_row(run_data, year):
    """Read a year row while accepting integer, float, or string JSON keys."""
    for key, value in run_data.items():
        try:
            if int(float(key)) == int(year):
                return value
        except (TypeError, ValueError):
            continue
    return None


def compute_frequency_from_saved(share_runs, year):
    """Compute dominant-technology frequencies from saved share trajectories."""
    counts = {technology: 0 for technology in TECHNOLOGIES}
    counts["mix"] = 0
    valid_runs = 0

    for run_data in share_runs:
        row = _get_year_row(run_data, year)
        if row is None:
            continue
        counts[get_dominance_state(row)] += 1
        valid_runs += 1

    if valid_runs == 0:
        raise ValueError(f"No runs contain target_year={year}")
    return {state: count / valid_runs for state, count in counts.items()}


def get_scenario_for_label(label_value):
    return "slow transition" if label_value == "020502" else "fast transition"


def compute_lp_fixed_frequency(label_value):
    """Return the saved LP Fixed dominant state for a paper scenario."""
    dominant_technology = {
        "122201": "solar pv electricity",
        "011301": "solar pv electricity",
        "020503": "hydroelectricity",
        "020502": "solar pv electricity",
    }.get(label_value)
    if dominant_technology is None:
        raise ValueError(f"No LP Fixed dominant technology for label={label_value}")

    frequency = {technology: 0.0 for technology in TECHNOLOGIES}
    frequency["mix"] = 0.0
    frequency[dominant_technology] = 1.0
    return frequency


def compute_dps_frequency(label_value, year):
    """Load saved DPS shares and compute their dominance frequency."""
    scenario_name = get_scenario_for_label(label_value)
    path = f"results/figures/{label_value}/shares_policy_{label_value}_{scenario_name}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing DPS share file: {path}")

    with open(path, "r") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return compute_frequency_from_saved(payload, year)
    if isinstance(payload, dict):
        row = _get_year_row(payload, year)
        if row is None:
            raise ValueError(f"No year={year} in {path}")
        frequency = {technology: 0.0 for technology in TECHNOLOGIES}
        frequency["mix"] = 0.0
        frequency[get_dominance_state(row)] = 1.0
        return frequency
    raise ValueError(f"Unsupported JSON structure in {path}")


def compute_lp_myopic_frequency(label_value, year):
    """Load the LP frequency stored in a dominance-frequency JSON file."""
    scenario_name = get_scenario_for_label(label_value)
    path = (
        f"results/figures/{label_value}/"
        f"dominance_freq_{label_value}_{scenario_name}_{year}.json"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing LP Myopic frequency file: {path}")

    with open(path, "r") as file:
        payload = json.load(file)
    lp_frequency = payload.get("LP")
    if lp_frequency is None:
        raise ValueError(f"Key 'LP' not found in {path}")

    frequency = {
        technology: float(lp_frequency.get(technology, 0.0))
        for technology in TECHNOLOGIES
    }
    frequency["mix"] = float(lp_frequency.get("mix", 0.0))
    return frequency


def load_combined_frequency_df(year, include_fixed=True):
    """Load all four scenarios into the table used by the combined plot."""
    states = TECHNOLOGIES + ["mix"]
    methods = ["LP Myopic", "DPS"]
    if include_fixed:
        methods = ["LP Fixed"] + methods

    rows = []
    for label_value, scenario_display_name in zip(COMPARISON_LABELS, COMPARISON_NAMES):
        try:
            if include_fixed:
                rows.append(
                    {
                        "ScenarioName": scenario_display_name,
                        "Mode": "LP Fixed",
                        **compute_lp_fixed_frequency(label_value),
                    }
                )
            rows.append(
                {
                    "ScenarioName": scenario_display_name,
                    "Mode": "LP Myopic",
                    **compute_lp_myopic_frequency(label_value, year),
                }
            )
            rows.append(
                {
                    "ScenarioName": scenario_display_name,
                    "Mode": "DPS",
                    **compute_dps_frequency(label_value, year),
                }
            )
        except Exception as error:
            print(f"Skipping label={label_value}: {error}")

    if not rows:
        return None, states, methods

    dataframe = pd.DataFrame(rows)
    dataframe["ScenarioName"] = pd.Categorical(
        dataframe["ScenarioName"],
        categories=COMPARISON_NAMES,
        ordered=True,
    )
    dataframe["Mode"] = pd.Categorical(
        dataframe["Mode"],
        categories=methods,
        ordered=True,
    )
    dataframe = dataframe.sort_values(["ScenarioName", "Mode"])
    return dataframe, states, methods


def print_frequency_summary(dataframe, states, year):
    """Print the frequency values used in the combined plot."""
    print(f"Dominant-technology frequency summary for {year}:")
    for _, row in dataframe.iterrows():
        values = ", ".join(
            f"{STATE_LABELS[state]}={100.0 * float(row[state]):.2f}%"
            for state in states
        )
        print(f"{row['ScenarioName']} | {row['Mode']} | {values}")


def plot_combined_frequency_bar(year, include_fixed=True):
    """Create the four-scenario stacked dominance-frequency figure."""
    dataframe, states, methods = load_combined_frequency_df(year, include_fixed)
    if dataframe is None:
        print("No combined frequency data are available to plot")
        return

    print_frequency_summary(dataframe, states, year)
    group_spacing = 1.12 if include_fixed else 1.06
    group_centers = np.arange(len(COMPARISON_NAMES)) * group_spacing
    method_count = len(methods)
    width = 0.24 if method_count == 3 else 0.26
    step = width + 0.06
    offsets = np.linspace(
        -(method_count - 1) * step / 2.0,
        (method_count - 1) * step / 2.0,
        method_count,
    )
    method_positions = {
        method: group_centers + offsets[index]
        for index, method in enumerate(methods)
    }
    method_labels = {
        "LP Fixed": "LP Fixed",
        "LP Myopic": "LP Myopic",
        "DPS": "DPS Adaptive",
    }

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    bottoms = {method: np.zeros(len(COMPARISON_NAMES)) for method in methods}
    legend_labels = set()

    for state in states:
        for method in methods:
            values = []
            for scenario_name in COMPARISON_NAMES:
                subset = dataframe[
                    (dataframe["ScenarioName"] == scenario_name)
                    & (dataframe["Mode"] == method)
                ]
                values.append(
                    100.0 * float(subset[state].iloc[0]) if not subset.empty else 0.0
                )
            values = np.asarray(values)
            display_label = STATE_LABELS[state]
            ax.bar(
                method_positions[method],
                values,
                bottom=bottoms[method],
                width=width,
                color=COLORS[state],
                edgecolor="black",
                linewidth=0.35,
                alpha=0.95,
                label=display_label if display_label not in legend_labels else None,
            )
            legend_labels.add(display_label)
            bottoms[method] += values

    ticks = []
    tick_labels = []
    for scenario_index in range(len(COMPARISON_NAMES)):
        for method in methods:
            ticks.append(method_positions[method][scenario_index])
            tick_labels.append(method_labels[method])
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=12, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Frequency of Dominant Tech (%)", fontsize=16)
    ax.tick_params(axis="x", pad=2)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    panel_label = "a" if year == 2050 else "b" if year == 2070 else None
    if panel_label:
        ax.text(
            0.01,
            0.99,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=20,
            fontweight="bold",
        )

    for index, scenario_name in enumerate(COMPARISON_NAMES):
        ax.text(
            group_centers[index],
            1.02,
            scenario_name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=13.5,
            clip_on=False,
        )

    ax.legend(
        title="Tech",
        fontsize=14,
        title_fontsize=14,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    suffix = "with_lp_det" if include_fixed else "no_lp_det"
    output_path = f"results/figures/dominance_stacked_cobar_{year}_{suffix}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined dominance-frequency plot saved to {output_path}")


def plot_frequency_bar(lp_frequency, dps_frequency, output_path, scenario_name):
    """Create the LP-versus-DPS stacked bar for the selected label."""
    states = [
        technology
        for technology in TECHNOLOGIES
        if technology != "SMR2 electricity" or label == "011301"
    ] + ["mix"]
    x_positions = np.array([0, 1])
    bottoms = np.zeros(2)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for state in states:
        values = np.array(
            [
                100.0 * lp_frequency.get(state, 0.0),
                100.0 * dps_frequency.get(state, 0.0),
            ]
        )
        ax.bar(
            x_positions,
            values,
            width=0.6,
            bottom=bottoms,
            color=COLORS[state],
            edgecolor="white",
            linewidth=0.7,
            label=STATE_LABELS[state],
            alpha=0.95,
        )
        bottoms += values

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["LP", "DPS"], fontsize=13, weight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Frequency (%)", fontsize=14, weight="bold")
    ax.set_title(
        f"Dominant Technology Frequency ({label}, {scenario_name}, {target_year})",
        fontsize=14,
        weight="bold",
    )
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Dominance-frequency plot saved to {output_path}")


def main():
    module_name, scenario_name = resolve_label_config(label)
    save_dir = f"results/figures/{label}"
    os.makedirs(save_dir, exist_ok=True)

    policy_shares_path = f"{save_dir}/shares_policy_{label}_{scenario_name}.json"
    lp_shares_path = f"{save_dir}/shares_exogenous_{label}_{scenario_name}.json"
    frequency_path = (
        f"{save_dir}/dominance_freq_{label}_{scenario_name}_{target_year}.json"
    )
    figure_path = (
        f"{save_dir}/dominance_freq_bar_{label}_{scenario_name}_{target_year}.png"
    )

    if simulate:
        lp_files = get_lp_files_for_runs(label, nsim)
        policy_shares = run_policy_shares(module_name, scenario_name)
        lp_shares = run_exogenous_shares(module_name, scenario_name, lp_files)
        with open(policy_shares_path, "w") as file:
            json.dump(policy_shares, file)
        with open(lp_shares_path, "w") as file:
            json.dump(lp_shares, file)
        print(f"Policy shares saved to {policy_shares_path}")
        print(f"LP shares saved to {lp_shares_path}")

    if simulate or plot_bar:
        if not simulate:
            missing = [
                path
                for path in (policy_shares_path, lp_shares_path)
                if not os.path.exists(path)
            ]
            if missing:
                raise FileNotFoundError(
                    f"Missing share files: {missing}. Set simulate=True first."
                )
            with open(policy_shares_path, "r") as file:
                policy_shares = json.load(file)
            with open(lp_shares_path, "r") as file:
                lp_shares = json.load(file)

        dps_frequency = compute_frequency_from_saved(policy_shares, target_year)
        lp_frequency = compute_frequency_from_saved(lp_shares, target_year)
        with open(frequency_path, "w") as file:
            json.dump({"LP": lp_frequency, "DPS": dps_frequency}, file, indent=2)
        print(f"Dominance frequencies saved to {frequency_path}")

        if plot_bar:
            plot_frequency_bar(lp_frequency, dps_frequency, figure_path, scenario_name)

    if plot_combined_bar:
        plot_combined_frequency_bar(target_year, include_fixed=include_lp_fixed)
        for year in summary_years:
            if int(year) == int(target_year):
                continue
            dataframe, states, _ = load_combined_frequency_df(year, include_lp_fixed)
            if dataframe is not None:
                print_frequency_summary(dataframe, states, year)

    if not simulate and not plot_bar and not plot_combined_bar:
        print("Nothing to do")


if __name__ == "__main__":
    main()
