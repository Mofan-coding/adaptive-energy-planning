"""Generate figure 2 for the paper.

"""

import importlib
import importlib.util
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import energySim._energy_sim_params as _energy_sim_params

# ===================== USER CONFIG =====================
simulate = False
plot = False
plot_co = True
plot_one = False

label = "020502" #122201; 011301; 020503
scenario = ""  # "" means auto-infer from label suffix ('-s' => slow transition)
nsim = 500
seed = 0

title = "Total Transition Cost Distribution"
gt_clip = 1.0
hidden_size = 2
input_norm = False
cost_same_as_lp = True
discount_rate = 0.02
trim_outlier_for_plot = True
outlier_quantile = 0.99


use_lp_folder = True

lp_folder = ""
# ======================================================


CSV_LP_METHOD = "LP (Determinstic)"
CSV_DPS_METHOD = "DPS (Adaptive)"

COMBINED_STRATEGY_LP_FIXED = "LP Fixed"
COMBINED_STRATEGY_LP_MYOPIC = "LP Myopic"
COMBINED_STRATEGY_DPS_ADAPTIVE = "DPS Adaptive"


def resolve_label_config(label):
    if label == "122201" or label == '022601' or label == '020503' or label == '020502':
        return "energySim.energy_sim_model", "1_results_lp_trajectories.json"
    if label == "011301":
        return "energySim.22_energy_sim_model", "2_results_lp_trajectories.json"
    raise ValueError(
        f"label={label} is not configured yet. "
        "Currently only '122201' and '011301' are enabled."
    )


def infer_scenario(label, scenario_arg):
    if scenario_arg:
        return scenario_arg
    base_label = label[:-2] if label.endswith("-s") else label
    return "slow transition" if base_label == "020502" else "fast transition"


def get_policy_path(label, scenario):
    # Keep same naming convention used in existing runs.
    scenario_for_policy = "fast transition"
    if label == "020502":
        scenario_for_policy = "slow transition"
    return f"results/{label}_{scenario_for_policy}_policy.pth"

def get_lp_folder_for_label(label):
    if label == "011301":
        return "results/_11_lp_nsim_runs"
    elif label == '020503':
        return "results/_22_lp_nsim_runs"
    elif label == '020502':
        return "results/_33_lp_nsim_runs"
    else:
        return "results/_00_lp_nsim_runs"

def get_lp_files_for_runs(label, nsim):
    _, single_lp_file = resolve_label_config(label)
    if not use_lp_folder:
        return [single_lp_file for _ in range(nsim)]

    folder = lp_folder if lp_folder else get_lp_folder_for_label(label)
    files = []
    for i in range(1, nsim + 1):
        p = os.path.join(folder, f"sim_{i:04d}_results_lp_trajectory.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing LP sim file: {p}")
        files.append(p)
    return files

def load_energy_model_module(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception:
        # Fallback for non-standard module names such as `22_energy_sim_model.py`.
        if module_name == "energySim.22_energy_sim_model":
            path = os.path.join("energySim", "22_energy_sim_model.py")
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        raise


def discount_tag(r):
    # e.g. 0.01 -> dis1, 0.02 -> dis2
    return f"dis{int(round(float(r) * 100))}"


def simulate_costs(label, scenario, nsim, seed, gt_clip, hidden_size, input_norm):
    module_name, lp_traj_file = resolve_label_config(label)
    _energy_sim_model = load_energy_model_module(module_name)
    lp_files = get_lp_files_for_runs(label, nsim)

    model = _energy_sim_model.EnergyModel(
        EFgp=_energy_sim_params.scenarios[scenario][0],
        slack=_energy_sim_params.scenarios[scenario][1],
        costparams=_energy_sim_params.costsAssumptions["Way et al. (2022)"],
        gt_clip=gt_clip,
        hidden_size=hidden_size,
        input_norm=input_norm,
        cost_same_as_lp=cost_same_as_lp,
        lp_trajectory_file=lp_traj_file,
        discount_rate=discount_rate,
    )

    costs_lp = []
    costs_dps = []

    model.mode = "exogenous"
    np.random.seed(seed)
    for i in range(nsim):
        # LP(sim i): load the corresponding LP trajectory/cost file.
        model.lp_trajectory_file = lp_files[i]
        model.load_lp_trajectories_from_file(lp_files[i])
        costs_lp.append(1e-12 * model.simulate())

    model.mode = "policy"
    policy_path = get_policy_path(label, scenario)
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    model.policy.load(policy_path)

    np.random.seed(seed)
    for i in range(nsim):
        if cost_same_as_lp:
            # DPS(sim i): use LP sim i for aligned shocks/learning params.
            model.lp_trajectory_file = lp_files[i]
            model.load_lp_trajectories_from_file(lp_files[i])
        else:
            # DPS uses original stochastic sampling; no LP shock/cost alignment.
            model.lp_trajectory_file = lp_traj_file
        costs_dps.append(1e-12 * model.simulate())

    return costs_lp, costs_dps


def build_plot_df(costs_lp, costs_dps):
    lp_df = pd.DataFrame({"Net Present Cost [trillion USD]": costs_lp})
    lp_df["Method"] = CSV_LP_METHOD

    dps_df = pd.DataFrame({"Net Present Cost [trillion USD]": costs_dps})
    dps_df["Method"] = CSV_DPS_METHOD

    return pd.concat([lp_df, dps_df], ignore_index=True)


def trim_outlier_by_quantile(df, q=0.99):
    # Trim long tails per method so one side does not dominate violin scale.
    kept = []
    for m in df["Method"].unique():
        d = df[df["Method"] == m].copy()
        ub = d["Net Present Cost [trillion USD]"].quantile(q)
        d = d[d["Net Present Cost [trillion USD]"] <= ub]
        kept.append(d)
    return pd.concat(kept, ignore_index=True)


def trim_outlier_by_quantile_grouped(df, q=0.99):
    # Trim outliers within each Scenario x Strategy group for fair side-by-side comparison.
    kept = []
    grouped = df.groupby(["Scenario", "Strategy"], dropna=False)
    for (_, _), d in grouped:
        ub = d["Net Present Cost [trillion USD]"].quantile(q)
        kept.append(d[d["Net Present Cost [trillion USD]"] <= ub].copy())
    return pd.concat(kept, ignore_index=True)


def make_plot(df, title):
    palette = {
        CSV_LP_METHOD: "#F58518",
        CSV_DPS_METHOD: "#4C78A8",
    }
    methods = [CSV_LP_METHOD, CSV_DPS_METHOD]

    plt.figure(figsize=(12, 6), dpi=300)
    ax = plt.gca()

    vp = sns.violinplot(
        data=df,
        x="Method",
        y="Net Present Cost [trillion USD]",
        inner=None,
        cut=0,
        width=0.5,
        palette=palette,
        order=methods,
    )
    for c in vp.collections:
        c.set_alpha(0.6)

    data_for_box = [
        df.loc[df["Method"] == m, "Net Present Cost [trillion USD]"] for m in methods
    ]
    bp = ax.boxplot(
        data_for_box,
        positions=[0, 1],
        widths=0.18,
        patch_artist=True,
        showfliers=False,
    )
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(palette[m])
        patch.set_alpha(0.9)
        patch.set_linewidth(2.0)
    for whisker in bp["whiskers"]:
        whisker.set_linewidth(2.0)
    for cap in bp["caps"]:
        cap.set_linewidth(2.0)
    for median in bp["medians"]:
        median.set_linewidth(2.4)
        median.set_color("black")

    sns.stripplot(
        data=df,
        x="Method",
        y="Net Present Cost [trillion USD]",
        color="black",
        size=3,
        alpha=0.35,
        jitter=0.08,
        order=methods,
    )

    ax.set_ylabel("Net Present Cost (Trillion USD)", fontsize=16, weight="bold")
    ax.set_xlabel("")
    ax.set_title(title, fontsize=18, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    ymin = 0
    ymax = float(df["Net Present Cost [trillion USD]"].max()) * 1.05
    ax.set_ylim(ymin, ymax)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()


def load_combined_cost_df(discount_rate):
    dtag = discount_tag(discount_rate)
    scenarios_cfg = [
        ("Base", "122201", "fast transition"),
        ("2 Breakthroughs", "011301", "fast transition"),
        ("Risk Averse", "020503", "fast transition"),
        ("Slow Transition", "020502", "slow transition"),
    ]
    rows = []

    for scen_name, lab, scen in scenarios_cfg:
        out_dir = f"results/figures/{lab}"
        csv_det = f"{out_dir}/{scen}_cost_violin_data.csv"
        csv_adp = f"{out_dir}/{scen}_1cost_violin_data_{dtag}.csv"

        if not os.path.exists(csv_det):
            raise FileNotFoundError(f"Missing CSV for {scen_name} LP deterministic: {csv_det}")
        if not os.path.exists(csv_adp):
            raise FileNotFoundError(f"Missing CSV for {scen_name} adaptive costs: {csv_adp}")

        det_df = pd.read_csv(csv_det)
        adp_df = pd.read_csv(csv_adp)
        required_cols = {"Net Present Cost [trillion USD]", "Method"}
        if not required_cols.issubset(set(det_df.columns)):
            raise ValueError(f"CSV missing required columns: {csv_det}")
        if not required_cols.issubset(set(adp_df.columns)):
            raise ValueError(f"CSV missing required columns: {csv_adp}")

        lp_det = det_df[det_df["Method"] == CSV_LP_METHOD].copy()
        lp_det["Scenario"] = scen_name
        lp_det["Strategy"] = COMBINED_STRATEGY_LP_FIXED
        rows.append(lp_det[["Net Present Cost [trillion USD]", "Scenario", "Strategy"]])

        lp_adp = adp_df[adp_df["Method"] == CSV_LP_METHOD].copy()
        lp_adp["Scenario"] = scen_name
        lp_adp["Strategy"] = COMBINED_STRATEGY_LP_MYOPIC
        rows.append(lp_adp[["Net Present Cost [trillion USD]", "Scenario", "Strategy"]])

        dps_adp = adp_df[adp_df["Method"] == CSV_DPS_METHOD].copy()
        dps_adp["Scenario"] = scen_name
        dps_adp["Strategy"] = COMBINED_STRATEGY_DPS_ADAPTIVE
        rows.append(dps_adp[["Net Present Cost [trillion USD]", "Scenario", "Strategy"]])

    return pd.concat(rows, ignore_index=True)


def load_base_case_cost_df(discount_rate):
    dtag = discount_tag(discount_rate)
    out_dir = "results/figures/122201"
    csv_det = f"{out_dir}/fast transition_cost_violin_data.csv"
    csv_adp = f"{out_dir}/fast transition_1cost_violin_data_{dtag}.csv"

    if not os.path.exists(csv_det):
        raise FileNotFoundError(f"Missing CSV for Base case LP fixed: {csv_det}")
    if not os.path.exists(csv_adp):
        raise FileNotFoundError(f"Missing CSV for Base case adaptive costs: {csv_adp}")

    det_df = pd.read_csv(csv_det)
    adp_df = pd.read_csv(csv_adp)
    required_cols = {"Net Present Cost [trillion USD]", "Method"}
    if not required_cols.issubset(set(det_df.columns)):
        raise ValueError(f"CSV missing required columns: {csv_det}")
    if not required_cols.issubset(set(adp_df.columns)):
        raise ValueError(f"CSV missing required columns: {csv_adp}")

    rows = []

    lp_fixed = det_df[det_df["Method"] == CSV_LP_METHOD].copy()
    lp_fixed["MethodLabel"] = "LP Fixed"
    rows.append(lp_fixed[["Net Present Cost [trillion USD]", "MethodLabel"]])

    lp_myopic = adp_df[adp_df["Method"] == CSV_LP_METHOD].copy()
    lp_myopic["MethodLabel"] = "LP Myopic"
    rows.append(lp_myopic[["Net Present Cost [trillion USD]", "MethodLabel"]])

    adaptive = adp_df[adp_df["Method"] == CSV_DPS_METHOD].copy()
    adaptive["MethodLabel"] = "Adaptive"
    rows.append(adaptive[["Net Present Cost [trillion USD]", "MethodLabel"]])

    return pd.concat(rows, ignore_index=True)


def make_combined_plot(df, title):
    scenarios_order = ["Base", "2 Breakthroughs", "Risk Averse", "Slow Transition"]
    strategy_order = [
        COMBINED_STRATEGY_LP_FIXED,
        COMBINED_STRATEGY_LP_MYOPIC,
        COMBINED_STRATEGY_DPS_ADAPTIVE,
    ]
    palette = {
        COMBINED_STRATEGY_LP_FIXED: "#f9d994",
        COMBINED_STRATEGY_LP_MYOPIC: "#edb200",
        COMBINED_STRATEGY_DPS_ADAPTIVE: "#cad5f2",
    }

    plt.figure(figsize=(12, 6), dpi=300)
    ax = plt.gca()

    vp = sns.violinplot(
        data=df,
        x="Scenario",
        y="Net Present Cost [trillion USD]",
        hue="Strategy",
        order=scenarios_order,
        hue_order=strategy_order,
        inner=None,
        cut=0,
        width=0.75,
        dodge=True,
        palette=palette,
    )
    for c in vp.collections:
        c.set_alpha(0.6)

    # Place boxplots at exact violin centers to guarantee visual alignment.
    def _extract_violin_centers(axis, n_expected):
        centers = []
        for coll in axis.collections:
            paths = coll.get_paths()
            if not paths:
                continue
            verts = paths[0].vertices
            if verts is None or len(verts) == 0:
                continue
            xs = verts[:, 0]
            if len(xs) == 0:
                continue
            centers.append(float((np.min(xs) + np.max(xs)) * 0.5))
        centers = sorted([c for c in centers if np.isfinite(c)])
        uniq = []
        tol = 1e-3
        for c in centers:
            if not uniq or abs(c - uniq[-1]) > tol:
                uniq.append(c)
        if len(uniq) == n_expected:
            return uniq
        return None

    n_expected = len(scenarios_order) * len(strategy_order)
    violin_centers = _extract_violin_centers(ax, n_expected)

    if violin_centers is None:
        fallback_offsets = {
            COMBINED_STRATEGY_LP_FIXED: -0.24,
            COMBINED_STRATEGY_LP_MYOPIC: 0.00,
            COMBINED_STRATEGY_DPS_ADAPTIVE: 0.24,
        }
        center_map = {}
        for i, scen in enumerate(scenarios_order):
            for s in strategy_order:
                center_map[(scen, s)] = i + fallback_offsets[s]
    else:
        center_map = {}
        k = 0
        for scen in scenarios_order:
            for s in strategy_order:
                center_map[(scen, s)] = violin_centers[k]
                k += 1

    positions = []
    data_for_box = []
    box_colors = []
    for scen in scenarios_order:
        for s in strategy_order:
            d = df.loc[
                (df["Scenario"] == scen) & (df["Strategy"] == s),
                "Net Present Cost [trillion USD]",
            ]
            if d.empty:
                continue
            positions.append(center_map[(scen, s)])
            data_for_box.append(d)
            box_colors.append(palette[s])

    if data_for_box:
        bp = ax.boxplot(
            data_for_box,
            positions=positions,
            widths=0.16,
            patch_artist=True,
            showfliers=False,
            zorder=3,
        )
        for patch, col in zip(bp["boxes"], box_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.9)
            patch.set_linewidth(1.6)
        for whisker in bp["whiskers"]:
            whisker.set_linewidth(1.2)
        for cap in bp["caps"]:
            cap.set_linewidth(1.2)
        for median in bp["medians"]:
            median.set_linewidth(1.8)
            median.set_color("black")

    ax.set_ylabel("Net Present Cost (Trillion USD)", fontsize=16)
    ax.set_xlabel("")
    #ax.set_title(title, fontsize=18, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(bottom=0)
    x_centers = np.arange(len(scenarios_order))
    ax.set_xticks(x_centers)
    ax.set_xticklabels(scenarios_order)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=palette[s], edgecolor="black", label=s) for s in strategy_order]
    ax.legend(legend_handles, strategy_order, loc="upper right", frameon=False, fontsize=16)

    plt.tight_layout()


def make_base_case_plot(df, title):
    method_order = ["LP Fixed", "LP Myopic", "Adaptive"]
    palette = {
        "LP Fixed": "#f9d994",
        "LP Myopic": "#edb200",
        "Adaptive": "#cad5f2",
    }

    plt.figure(figsize=(12, 6), dpi=300)
    ax = plt.gca()

    vp = sns.violinplot(
        data=df,
        x="MethodLabel",
        y="Net Present Cost [trillion USD]",
        inner=None,
        cut=0,
        width=0.65,
        palette=palette,
        order=method_order,
    )
    for c in vp.collections:
        c.set_alpha(0.6)

    data_for_box = [
        df.loc[df["MethodLabel"] == m, "Net Present Cost [trillion USD]"] for m in method_order
    ]
    bp = ax.boxplot(
        data_for_box,
        positions=np.arange(len(method_order)),
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        zorder=3,
    )
    for patch, m in zip(bp["boxes"], method_order):
        patch.set_facecolor(palette[m])
        patch.set_alpha(0.9)
        patch.set_linewidth(1.6)
    for whisker in bp["whiskers"]:
        whisker.set_linewidth(1.2)
    for cap in bp["caps"]:
        cap.set_linewidth(1.2)
    for median in bp["medians"]:
        median.set_linewidth(1.8)
        median.set_color("black")

    ax.set_ylabel("Net Present Cost (Trillion USD)", fontsize=20)
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_xticks(np.arange(len(method_order)))
    ax.set_xticklabels(method_order, fontsize=18)
    ax.set_ylim(bottom=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()


def print_p99_cost_reduction_summary(df):
    scenarios_order = ["Base", "2 Breakthroughs", "Risk Averse", "Slow Transition"]
    strategy_order = [
        COMBINED_STRATEGY_LP_FIXED,
        COMBINED_STRATEGY_LP_MYOPIC,
        COMBINED_STRATEGY_DPS_ADAPTIVE,
    ]
    cost_col = "Net Present Cost [trillion USD]"

    print("\n=== 99th Percentile Cost Summary ===")
    det_reductions = []
    adp_reductions = []
    median_det_reductions = []
    median_adp_reductions = []
    lower_det_reductions = []
    lower_adp_reductions = []

    for scen in scenarios_order:
        sub = df[df["Scenario"] == scen].copy()
        if sub.empty:
            print(f"{scen}: no data, skip.")
            continue

        p99 = {}
        median = {}
        lower = {}
        missing = []
        for strategy in strategy_order:
            vals = sub.loc[sub["Strategy"] == strategy, cost_col].dropna().to_numpy()
            if len(vals) == 0:
                missing.append(strategy)
                continue
            p99[strategy] = float(np.percentile(vals, 99))
            median[strategy] = float(np.percentile(vals, 50))
            lower[strategy] = float(np.percentile(vals, 1))

        if missing:
            print(f"{scen}: missing strategies {missing}, skip.")
            continue

        red_vs_det = (
            100.0
            * (p99[COMBINED_STRATEGY_LP_FIXED] - p99[COMBINED_STRATEGY_DPS_ADAPTIVE])
            / p99[COMBINED_STRATEGY_LP_FIXED]
        )
        red_vs_adp = (
            100.0
            * (p99[COMBINED_STRATEGY_LP_MYOPIC] - p99[COMBINED_STRATEGY_DPS_ADAPTIVE])
            / p99[COMBINED_STRATEGY_LP_MYOPIC]
        )
        det_reductions.append(red_vs_det)
        adp_reductions.append(red_vs_adp)

        median_red_vs_det = (
            100.0
            * (median[COMBINED_STRATEGY_LP_FIXED] - median[COMBINED_STRATEGY_DPS_ADAPTIVE])
            / median[COMBINED_STRATEGY_LP_FIXED]
        )
        median_red_vs_adp = (
            100.0
            * (median[COMBINED_STRATEGY_LP_MYOPIC] - median[COMBINED_STRATEGY_DPS_ADAPTIVE])
            / median[COMBINED_STRATEGY_LP_MYOPIC]
        )
        lower_red_vs_det = (
            100.0
            * (lower[COMBINED_STRATEGY_LP_FIXED] - lower[COMBINED_STRATEGY_DPS_ADAPTIVE])
            / lower[COMBINED_STRATEGY_LP_FIXED]
        )
        lower_red_vs_adp = (
            100.0
            * (lower[COMBINED_STRATEGY_LP_MYOPIC] - lower[COMBINED_STRATEGY_DPS_ADAPTIVE])
            / lower[COMBINED_STRATEGY_LP_MYOPIC]
        )
        median_det_reductions.append(median_red_vs_det)
        median_adp_reductions.append(median_red_vs_adp)
        lower_det_reductions.append(lower_red_vs_det)
        lower_adp_reductions.append(lower_red_vs_adp)

        print(
            f"{scen}: "
            f"p99(LP fixed)={p99[COMBINED_STRATEGY_LP_FIXED]:.2f}, "
            f"p99(LP myopic)={p99[COMBINED_STRATEGY_LP_MYOPIC]:.2f}, "
            f"p99(DPS)={p99[COMBINED_STRATEGY_DPS_ADAPTIVE]:.2f}, "
            f"DPS lower by {red_vs_det:.2f}% vs LP fixed, "
            f"{red_vs_adp:.2f}% vs LP myopic."
        )

        print(
            f"{scen}: "
            f"median(LP fixed)={median[COMBINED_STRATEGY_LP_FIXED]:.2f}, "
            f"median(LP myopic)={median[COMBINED_STRATEGY_LP_MYOPIC]:.2f}, "
            f"median(DPS)={median[COMBINED_STRATEGY_DPS_ADAPTIVE]:.2f}, "
            f"DPS lower by {median_red_vs_det:.2f}% vs LP fixed, "
            f"{median_red_vs_adp:.2f}% vs LP myopic."
        )

        print(
            f"{scen}: "
            f"p1 lower end(LP fixed)={lower[COMBINED_STRATEGY_LP_FIXED]:.2f}, "
            f"p1 lower end(LP myopic)={lower[COMBINED_STRATEGY_LP_MYOPIC]:.2f}, "
            f"p1 lower end(DPS)={lower[COMBINED_STRATEGY_DPS_ADAPTIVE]:.2f}, "
            f"DPS lower by {lower_red_vs_det:.2f}% vs LP fixed, "
            f"{lower_red_vs_adp:.2f}% vs LP myopic."
        )

    if len(det_reductions) == 4 and len(adp_reductions) == 4:
        det_text = ", ".join(f"{v:.2f}%" for v in det_reductions)
        adp_text = ", ".join(f"{v:.2f}%" for v in adp_reductions)
        print("\nPaper-ready sentence numbers:")
        print(f"Relative to LP fixed benchmarks: {det_text}.")
        print(f"Relative to LP myopic benchmarks: {adp_text}.")

    if len(median_det_reductions) == 4 and len(median_adp_reductions) == 4:
        median_det_text = ", ".join(f"{v:.2f}%" for v in median_det_reductions)
        median_adp_text = ", ".join(f"{v:.2f}%" for v in median_adp_reductions)
        print("\nMedian value relative to benchmark value:")
        print(f"Relative to LP fixed benchmarks: {median_det_text}.")
        print(f"Relative to LP myopic benchmarks: {median_adp_text}.")

    if len(lower_det_reductions) == 4 and len(lower_adp_reductions) == 4:
        lower_det_text = ", ".join(f"{v:.2f}%" for v in lower_det_reductions)
        lower_adp_text = ", ".join(f"{v:.2f}%" for v in lower_adp_reductions)
        print("\nLower end of the distribution relative to benchmark value:")
        print(f"Relative to LP fixed benchmarks: {lower_det_text}.")
        print(f"Relative to LP myopic benchmarks: {lower_adp_text}.")


def main():
    run_scenario = infer_scenario(label, scenario)
    _, legacy_lp_file = resolve_label_config(label)
    print(f"label={label}, scenario={run_scenario}, nsim={nsim}")
    print(f"cost_same_as_lp={cost_same_as_lp}")
    if use_lp_folder:
        active_folder = lp_folder if lp_folder else get_lp_folder_for_label(label)
        print(f"LP source: folder mode -> {active_folder} (legacy single file ignored)")
    else:
        print(f"LP source: single-file mode -> {legacy_lp_file}")

    out_dir = f"results/figures/{label}"
    os.makedirs(out_dir, exist_ok=True)
    dtag = discount_tag(discount_rate)
    csv_path = f"{out_dir}/{run_scenario}_1cost_violin_data_{dtag}.csv"
    fig_path = f"{out_dir}/{run_scenario}_1cost_violin_paper_{dtag}.png"

    if simulate:
        costs_lp, costs_dps = simulate_costs(
            label=label,
            scenario=run_scenario,
            nsim=nsim,
            seed=seed,
            gt_clip=gt_clip,
            hidden_size=hidden_size,
            input_norm=input_norm,
        )
        plot_df = build_plot_df(costs_lp, costs_dps)
        plot_df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

    if plot:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CSV not found for plotting: {csv_path}. Set simulate=True first."
            )
        plot_df = pd.read_csv(csv_path)
        required_cols = {"Net Present Cost [trillion USD]", "Method"}
        if not required_cols.issubset(set(plot_df.columns)):
            raise ValueError(f"CSV missing required columns: {sorted(required_cols)}")

        plot_df_for_plot = plot_df
        if trim_outlier_for_plot:
            before_n = len(plot_df_for_plot)
            plot_df_for_plot = trim_outlier_by_quantile(plot_df_for_plot, q=outlier_quantile)
            after_n = len(plot_df_for_plot)
            print(
                f"Outlier trim for plotting: q={outlier_quantile}, "
                f"kept {after_n}/{before_n} points."
            )

        make_plot(plot_df_for_plot, title)
        plt.savefig(fig_path)
        plt.show()
        print(f"Saved figure: {fig_path}")

    if plot_co:
        combo_df = load_combined_cost_df(discount_rate=discount_rate)
        print_p99_cost_reduction_summary(combo_df)
        combo_df_for_plot = combo_df
        if trim_outlier_for_plot:
            before_n = len(combo_df_for_plot)
            combo_df_for_plot = trim_outlier_by_quantile_grouped(combo_df_for_plot, q=outlier_quantile)
            after_n = len(combo_df_for_plot)
            print(
                f"Outlier trim for combined plotting: q={outlier_quantile}, "
                f"kept {after_n}/{before_n} points."
            )

        make_combined_plot(
            combo_df_for_plot,
            "Total Transition Cost Distribution across Investment Strategies",
        )
        dtag = discount_tag(discount_rate)
        out_path = f"results/figures/combined_cost_violin_4scenarios_{dtag}.png"
        plt.savefig(out_path)
        plt.show()
        print(f"Saved combined figure: {out_path}")

    if plot_one:
        one_df = load_base_case_cost_df(discount_rate=discount_rate)
        one_df_for_plot = one_df
        if trim_outlier_for_plot:
            before_n = len(one_df_for_plot)
            kept = []
            for method_label in one_df_for_plot["MethodLabel"].unique():
                d = one_df_for_plot[one_df_for_plot["MethodLabel"] == method_label].copy()
                ub = d["Net Present Cost [trillion USD]"].quantile(outlier_quantile)
                kept.append(d[d["Net Present Cost [trillion USD]"] <= ub].copy())
            one_df_for_plot = pd.concat(kept, ignore_index=True)
            after_n = len(one_df_for_plot)
            print(
                f"Outlier trim for base-case plotting: q={outlier_quantile}, "
                f"kept {after_n}/{before_n} points."
            )

        make_base_case_plot(
            one_df_for_plot,
            "Total Transition Cost Distribution across Investment Strategies",
        )
        dtag = discount_tag(discount_rate)
        out_path = f"results/figures/base_case_cost_violin_3methods_{dtag}.png"
        plt.savefig(out_path)
        plt.show()
        print(f"Saved base-case figure: {out_path}")

    if not simulate and not plot and not plot_co and not plot_one:
        print("Nothing to do. Set simulate=True and/or plot=True.")


if __name__ == "__main__":
    main()
