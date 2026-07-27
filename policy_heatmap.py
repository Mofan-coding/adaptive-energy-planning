"""Generate the paper's policy-response heatmap (Figure 5).

The script loads a trained DPS policy, evaluates its growth-rate action across
a grid of technology unit costs and cumulative production, and plots the
response in interpretable raw units with decision contours.
"""

import importlib
import importlib.util
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import energySim._energy_sim_params as _energy_sim_params


# ===================== USER CONFIG =====================
label = "011301"
scenario = "fast transition"

gt_clip = 1.0
hidden_size = 2
input_norm = False

# Fixed policy inputs represented by this heatmap.
time_norm = 0.375
grid_balance_x10 = 1.2
tech_share = 0.12

# Ranges use the policy's transformed feature units:
# log10(unit cost) and log10(cumulative production) / 10.
cost_feature_range = (0.5, 2.0)
production_feature_range = (0.0, 0.35)
grid_size = 160

color_map = "Blues"
color_min = -0.1
color_max = 1.4
contour_levels = (0.2, 0.5, 1.0)
show_feature_axes = False
plot_title = "2"
# =======================================================


matplotlib.rc("savefig", dpi=300)
matplotlib.rc(
    "font",
    **{"family": "sans-serif", "sans-serif": "Helvetica"},
)


def resolve_model_module(label_value):
    """Return the final EnergySim implementation for a policy label."""
    if label_value in {"011301", "011903", "011903-s"}:
        return "energySim.22_energy_sim_model"
    return "energySim.energy_sim_model"


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


def load_policy(label_value, scenario_name):
    """Build the matching EnergySim model and load its trained policy."""
    module_name = resolve_model_module(label_value)
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

    policy_path = f"results/{label_value}_{scenario_name}_policy.pth"
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    model.policy.load(policy_path)
    return model.policy


def evaluate_policy_grid(
    policy,
    normalized_time,
    grid_balance,
    technology_share,
    cost_range,
    production_range,
    bins,
):
    """Evaluate policy actions on a raw cost-production grid."""
    cost_min = 10 ** float(cost_range[0])
    cost_max = 10 ** float(cost_range[1])
    production_min = 10 ** (10.0 * float(production_range[0]))
    production_max = 10 ** (10.0 * float(production_range[1]))

    unit_costs = np.logspace(np.log10(cost_min), np.log10(cost_max), bins)
    cumulative_production = np.logspace(
        np.log10(production_min),
        np.log10(production_max),
        bins,
    )
    actions = np.empty((bins, bins), dtype=float)

    for cost_index, unit_cost in enumerate(unit_costs):
        log_cost = np.log10(unit_cost)
        for production_index, production in enumerate(cumulative_production):
            production_feature = np.log10(production) / 10.0
            policy_input = np.array(
                [
                    log_cost,
                    production_feature,
                    normalized_time,
                    grid_balance,
                    technology_share,
                ],
                dtype=float,
            )
            action = policy.get_action(policy_input)
            actions[cost_index, production_index] = float(
                np.asarray(action).reshape(-1)[0]
            )

    return cumulative_production, unit_costs, actions


def plot_policy_heatmap(
    policy,
    normalized_time,
    grid_balance,
    technology_share,
    cost_range=(0.5, 2.0),
    production_range=(0.0, 0.35),
    bins=160,
    cmap="Blues",
    vmin=-0.1,
    vmax=1.4,
    contours=(0.2, 0.5, 1.0),
    show_transformed_axes=False,
    title=None,
    output_path=None,
):
    """Plot policy growth actions in raw cost and production units."""
    production, costs, actions = evaluate_policy_grid(
        policy=policy,
        normalized_time=normalized_time,
        grid_balance=grid_balance,
        technology_share=technology_share,
        cost_range=cost_range,
        production_range=production_range,
        bins=bins,
    )

    production_grid, cost_grid = np.meshgrid(production, costs)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    heatmap = ax.pcolormesh(
        production_grid,
        cost_grid,
        actions,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cumulative Production (EJ)", fontsize=18)
    ax.set_ylabel("Unit Cost (USD/GJ)", fontsize=18)
    if title:
        ax.set_title(title, fontsize=20, weight="bold")

    if contours:
        contour_plot = ax.contour(
            production_grid,
            cost_grid,
            actions,
            levels=list(contours),
            colors="black",
            linewidths=1.0,
            alpha=0.9,
        )
        ax.clabel(contour_plot, inline=True, fontsize=18, fmt="%.2g")

    colorbar = fig.colorbar(heatmap, ax=ax, pad=0.02, shrink=0.9)
    colorbar.set_label("Policy action (growth rate)")

    if show_transformed_axes:
        production_axis = ax.secondary_xaxis(
            "top",
            functions=(
                lambda value: np.log10(value) / 10.0,
                lambda feature: 10 ** (10.0 * feature),
            ),
        )
        production_axis.set_xlabel("log10(cumulative production) / 10")

        cost_axis = ax.secondary_yaxis(
            "right",
            functions=(lambda value: np.log10(value), lambda feature: 10**feature),
        )
        cost_axis.set_ylabel("log10(unit cost)")

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Policy heatmap saved to {output_path}")

    return fig, ax


def main():
    policy = load_policy(label, scenario)
    output_path = f"results/figures/{label}/heatmap2050_paper_original_domain.png"
    figure, _ = plot_policy_heatmap(
        policy=policy,
        normalized_time=time_norm,
        grid_balance=grid_balance_x10,
        technology_share=tech_share,
        cost_range=cost_feature_range,
        production_range=production_feature_range,
        bins=grid_size,
        cmap=color_map,
        vmin=color_min,
        vmax=color_max,
        contours=contour_levels,
        show_transformed_axes=show_feature_axes,
        title=plot_title,
        output_path=output_path,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
