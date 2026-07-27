"""
File: _11_run_annual_LP.py
Last Updated: 2026-02-25
Purpose:
- Annual LP rollout with stochastic costs (nsim runs), plus:
  1) non-decrease floor (net of retirement),
  2) lifetime retirement through vintage tracking.
- Two-SMR version (`SMR electricity` + `SMR2 electricity`).

# LP adaptive

Notes:
- This keeps the same output payload style as run_annual_LP.py.
- Retirement mode can be set to 'linear' (default, closer to sim model)
  or 'cliff' (hard retirement at end of lifetime).
"""

import json
import copy
from pathlib import Path

import numpy as np
import pyomo.environ as pyo

from energySim._energy_sim_params import scenarios, costparams


SMR_TECH = "SMR electricity"
SMR2_TECH = "SMR2 electricity"
SMR_SEED_YEAR = 2030
SMR_SEED_GEN = 0.058

# ===================== USER CONFIG =====================
nsim = 200
scenario = "fast transition"
nscenarios = 10
seed = 0
world_seed = 0
planner_seed = 1
outdir = "results/_11_lp_nsim_runs"
deterministic = False

forecast = False # if true, sample 100 to optimzie, if false, just use past cost relizations
risk_averse = False
risk_quantile = 0.90
# Planner omega knowledge switch:1
# True  -> planner uses tech-specific expected omega from params.
# False -> planner uses common prior omega.
omega = True
omega_prior_common = 0.2

retirement_mode = "linear"  # "linear" or "cliff"
default_lifetime = 60
min_cap = False  # True: gen_t >= previous_gen_t ; False: no LP tech-level minimum
max_cap = True  # True: gen_t <= 2*anchor ; False: no DPS-style growth cap
# =======================================================


def build_year_model():
    m = pyo.AbstractModel()
    m.TECHNOLOGIES = pyo.Set()
    m.SCENARIOS = pyo.Set()

    m.demand = pyo.Param()
    m.unit_cost = pyo.Param(m.TECHNOLOGIES, m.SCENARIOS)
    m.min_gen = pyo.Param(m.TECHNOLOGIES)
    m.max_gen = pyo.Param(m.TECHNOLOGIES)

    m.gen = pyo.Var(m.TECHNOLOGIES, domain=pyo.NonNegativeReals)

    def obj_fun(model):
        return sum(
            model.unit_cost[i, j] * model.gen[i]
            for i in model.TECHNOLOGIES
            for j in model.SCENARIOS
        ) / len(model.SCENARIOS)

    m.obj = pyo.Objective(rule=obj_fun)

    def demand_satisfaction(model):
        return sum(model.gen[i] for i in model.TECHNOLOGIES) == model.demand

    m.demand_constraints = pyo.Constraint(rule=demand_satisfaction)

    def lower_bound(model, i):
        return model.gen[i] >= model.min_gen[i]

    def upper_bound(model, i):
        return model.gen[i] <= model.max_gen[i]

    m.lb = pyo.Constraint(m.TECHNOLOGIES, rule=lower_bound)
    m.ub = pyo.Constraint(m.TECHNOLOGIES, rule=upper_bound)
    return m


def get_base_technologies(transition_name: str):
    base_technologies = []
    for el in scenarios[transition_name][0]:
        if (
            el[1] == "electricity"
            and el[0] not in ["EV batteries", "multi-day batteries", "P2Xfuels", "daily batteries"]
        ):
            base_technologies.append(el[0])
    for el in scenarios[transition_name][1]:
        if (
            el == "electricity"
            and scenarios[transition_name][1][el]
            not in ["EV batteries", "multi-day batteries", "P2Xfuels", "daily batteries"]
        ):
            base_technologies.append(scenarios[transition_name][1][el])
    return base_technologies


def build_init_params(base_technologies):
    init_unit_cost = {}
    init_cum_prod = {}
    init_gen = {
        "coal electricity": 35.7,
        "gas electricity": 22.9,
        "nuclear electricity": 10.0,
        "hydroelectricity": 15.2,
        "biopower electricity": 2.55,
        "wind electricity": 5.75,
        "solar pv electricity": 3.0,
    }
    lexp = {}
    sigma = {}
    mr = {}
    k = {}

    for tech in base_technologies:
        init_unit_cost[tech] = costparams["c0"][tech]
        init_cum_prod[tech] = costparams["z0"][tech]
        lexp[tech] = -costparams["omega"].get(tech, 0.0)
        sigma[tech] = costparams["sigma"].get(tech)
        mr[tech] = costparams["mr"].get(tech)
        k[tech] = costparams["k"].get(tech)

    init_unit_cost.setdefault(SMR_TECH, costparams["c0"].get(SMR_TECH, 20.0))
    init_cum_prod.setdefault(SMR_TECH, costparams["z0"].get(SMR_TECH, 1e-9))
    lexp.setdefault(SMR_TECH, -costparams["omega"].get(SMR_TECH, 0.074))
    sigma.setdefault(SMR_TECH, costparams["sigma"].get(SMR_TECH))
    mr.setdefault(SMR_TECH, costparams["mr"].get(SMR_TECH))
    k.setdefault(SMR_TECH, costparams["k"].get(SMR_TECH))
    init_gen.setdefault(SMR_TECH, 0.0)
    init_unit_cost.setdefault(SMR2_TECH, costparams["c0"].get(SMR2_TECH, 20.0))
    init_cum_prod.setdefault(SMR2_TECH, costparams["z0"].get(SMR2_TECH, 1e-9))
    lexp.setdefault(SMR2_TECH, -costparams["omega"].get(SMR2_TECH, 0.074))
    sigma.setdefault(SMR2_TECH, costparams["sigma"].get(SMR2_TECH))
    mr.setdefault(SMR2_TECH, costparams["mr"].get(SMR2_TECH))
    k.setdefault(SMR2_TECH, costparams["k"].get(SMR2_TECH))
    init_gen.setdefault(SMR2_TECH, 0.0)

    return init_unit_cost, init_cum_prod, init_gen, lexp, sigma, mr, k


def load_demand(scenario_name: str, init_gen):
    demand_path = Path(f"{scenario_name}_results_elec_demand.json")
    years = [x for x in range(2021, 2101)]
    demand = {}
    try:
        with demand_path.open("r") as f:
            elec_json = json.load(f)
        elec_demand_map = elec_json.get("elec_demand", {})
        demand_init = float(elec_demand_map["2020"])
        for y in years:
            key = str(y)
            if key in elec_demand_map:
                demand[y] = float(elec_demand_map[key])
            else:
                demand[y] = demand.get(y - 1, demand_init) * 1.02
        demand[2101] = float(elec_demand_map.get("2100", demand[2100])) * 1.02
        years.append(2101)
        print(f"Loaded demand from {demand_path}")
    except Exception as e:
        demand_init = sum(init_gen.values())
        for y in years:
            demand[y] = demand_init * (1.02) ** (y - years[0])
        demand[2101] = demand[2100] * 1.02
        years.append(2101)
        print(f"Fallback demand construction used: {e}")
    return years, demand


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, tuple):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def survival_factor(age, lifetime, mode):
    if lifetime <= 0:
        return 0.0
    if mode == "cliff":
        # Match sim-model carry-over: survives through age==L, retires after.
        return 1.0 if age <= lifetime else 0.0
    # linear retirement (for initial 2020 stock in sim-model vintage logic)
    if age < 0:
        return 0.0
    if age >= lifetime:
        return 0.0
    return max(0.0, 1.0 - age / float(lifetime))


def get_lifetime(tech):
    return int(costparams.get("L", {}).get(tech, default_lifetime))


def existing_surviving_capacity(vintages, year, tech, init_gen):
    life = get_lifetime(tech)
    total = 0.0
    # Sim-model-style: initial stock (year 2020) retires linearly.
    age0 = year - 2020
    total += init_gen.get(tech, 0.0) * survival_factor(age0, life, "linear")

    # Sim-model-style: post-2020 new builds retire with cliff lifetime.
    for v_year, v_amount in vintages.get(tech, []):
        if v_year >= year or v_year <= 2020:
            continue
        age = year - v_year
        total += v_amount * survival_factor(age, life, "cliff")
    return max(0.0, float(total))


def pre_sample_real_world_scenarios(nsim, years, techs_all, lexp, sigma, mr, seed_):
    """Pre-sample true-world omega and yearly shocks, independent of LP forecast settings."""
    sigma_omega = costparams.get("sigmaOmega", {})
    worlds = {}
    for sim_id in range(1, nsim + 1):
        rng_world = np.random.default_rng(seed_ + sim_id)

        lexp_real = {}
        for t in techs_all:
            mu_omega = costparams.get("omega", {}).get(t, None)
            if mr.get(t) is None and mu_omega is not None:
                sd_omega = sigma_omega.get(t, 0.0) or 0.0
                lexp_real[t] = -float(rng_world.normal(mu_omega, sd_omega))
            else:
                lexp_real[t] = lexp.get(t, 0.0)

        shocks_ar1 = {str(y): {"1": {}} for y in years}
        shocks_iid = {str(y): {"1": {}} for y in years}
        for y in years:
            for t in techs_all:
                if mr.get(t) is not None:
                    eps = float(rng_world.normal(0.0, sigma.get(t, 0.0) or 0.0))
                    shocks_ar1[str(y)]["1"][t] = eps
                else:
                    sd = (
                        np.sqrt((sigma.get(t, 0.0) ** 2) / (1 + 0.19**2))
                        if sigma.get(t) is not None
                        else 0.0
                    )
                    eps = float(rng_world.normal(0.0, sd))
                    shocks_iid[str(y)]["1"][t] = eps

        worlds[str(sim_id)] = {
            "learning_exponent": lexp_real,
            "shared_shocks": {
                "ar1_cost": shocks_ar1,
                "wright_iid": shocks_iid,
            },
        }
    return worlds


def run_one_sim(
    sim_id,
    years,
    demand,
    base_technologies,
    init_unit_cost,
    init_cum_prod,
    init_gen,
    lexp,
    sigma,
    mr,
    k,
    nscenarios,
    stochastic,
    rng_planner,
    world_sample,
):
    year_model = build_year_model()

    techs_all = list(base_technologies)
    if SMR_TECH not in techs_all:
        techs_all.append(SMR_TECH)
    if SMR2_TECH not in techs_all:
        techs_all.append(SMR2_TECH)

    cum_prod = {t: init_cum_prod[t] + init_gen.get(t, 0.0) for t in techs_all}
    generation_traj = {t: [init_gen.get(t, 0.0)] for t in techs_all}
    unit_cost_traj = {t: [init_unit_cost[t]] for t in techs_all}
    cum_gen_traj = {t: [init_cum_prod[t]] for t in techs_all}

    shocks_ar1 = copy.deepcopy(world_sample.get("shared_shocks", {}).get("ar1_cost", {}))
    shocks_iid = copy.deepcopy(world_sample.get("shared_shocks", {}).get("wright_iid", {}))

    # Real-world realized cost state (hidden omega, yearly real shocks).
    realized_cost = {t: init_unit_cost[t] for t in techs_all}
    real_dist = {t: 0.0 for t in techs_all}
    # Planner-inferred disturbance state from observed costs (uses expected lexp).
    planner_dist = {t: 0.0 for t in techs_all}
    previous_gen = {t: init_gen.get(t, 0.0) for t in techs_all}

    # Vintage bookkeeping for lifetime retirement (store only post-2020 builds).
    vintages = {t: [] for t in techs_all}

    # True-world learning exponent is pre-sampled per sim and fixed over years.
    lexp_real = world_sample.get("learning_exponent", {})

    for y in years:
        techs_y = list(base_technologies)
        if y >= 2030:
            techs_y.append(SMR_TECH)
            techs_y.append(SMR2_TECH)

        # min_cap controls the only LP lower bound behavior:
        # - True: non-decrease vs previous year
        # - False: no tech-level lower bound
        min_gen = {t: 0.0 for t in techs_y}
        if min_cap:
            for t in techs_y:
                min_gen[t] = max(min_gen[t], previous_gen.get(t, 0.0))
        # Match DPS growth upper bound gt<=1: q_{t+1} <= 2 * q_t.
        # Use previous_gen anchor (and guard feasibility with min_gen floor).
        max_gen = {}
        if max_cap:
            for t in techs_y:
                anchor = max(previous_gen.get(t, 0.0), min_gen[t])
                max_gen[t] = max(min_gen[t], 2.0 * anchor)
        else:
            # No per-tech growth cap; only bounded by system demand balance.
            for t in techs_y:
                max_gen[t] = max(min_gen[t], demand[y])

        param_file = Path("model_params") / f"param_11sim{sim_id:04d}_{y}.dat"
        with param_file.open("w") as f:
            f.write("set TECHNOLOGIES := \n")
            for t in techs_y:
                f.write(f"{t.replace(' ', '-')}\n")
            f.write(";\n\n")

            scen_count = nscenarios if forecast else 1
            f.write("set SCENARIOS := \n")
            for nscen in range(1, scen_count + 1):
                f.write(f"{nscen}\n")
            f.write(";\n\n")

            f.write("param unit_cost := \n")
            # Planner cost used in LP objective per technology.
            # - forecast=False: no scenario forecast; use observed realized_cost directly.
            # - forecast=True, risk_averse=False: use mean of forecast distribution.
            # - forecast=True, risk_averse=True: use upper quantile (e.g., P90).
            planner_cost = {}
            for t in techs_y:
                if not forecast:
                    planner_cost[t] = float(realized_cost[t])
                    continue

                omega_fore = (-lexp.get(t, 0.0)) if omega else omega_prior_common
                lexp_fore = -omega_fore

                samples = []
                for _ in range(scen_count):
                    if stochastic:
                        if mr.get(t) is not None:
                            eps_f = float(rng_planner.normal(0.0, sigma.get(t, 0.0) or 0.0))
                            c_fore = np.exp(
                                mr[t] * np.log(max(realized_cost[t], 1e-12))
                                + eps_f
                                + (k[t] if k.get(t) is not None else 0.0)
                            )
                        else:
                            sd = (
                                np.sqrt((sigma.get(t, 0.0) ** 2) / (1 + 0.19**2))
                                if sigma.get(t) is not None
                                else 0.0
                            )
                            eps_f = float(rng_planner.normal(0.0, sd))
                            dist_f = planner_dist.get(t, 0.0) * 0.19 + eps_f
                            z_prev = max(cum_prod[t] - generation_traj[t][-1], 1e-12)
                            z_curr = max(cum_prod[t], z_prev)
                            c_fore = np.exp(
                                np.log(max(realized_cost[t], 1e-12))
                                + lexp_fore * np.log(z_curr / z_prev)
                                + dist_f
                            )
                    else:
                        z0 = max(init_cum_prod[t], 1e-12)
                        zc = max(cum_prod[t], z0)
                        c_fore = init_unit_cost[t] * (zc / z0) ** lexp_fore
                    samples.append(float(c_fore))

                if risk_averse:
                    planner_cost[t] = float(np.quantile(samples, risk_quantile))
                else:
                    planner_cost[t] = float(np.mean(samples))

            for nscen in range(1, scen_count + 1):
                for t in techs_y:
                    f.write(f"{t.replace(' ', '-')} {nscen} {planner_cost[t]}\n")
            f.write(";\n\n")

            f.write("param min_gen := \n")
            for t in techs_y:
                f.write(f"{t.replace(' ', '-')} {min_gen[t]}\n")
            f.write(";\n\n")

            f.write("param max_gen := \n")
            for t in techs_y:
                f.write(f"{t.replace(' ', '-')} {max_gen[t]}\n")
            f.write(";\n\n")

            f.write("param demand := \n")
            f.write(f"{demand[y]}\n")
            f.write(";\n\n")

        instance = year_model.create_instance(str(param_file))
        solver = pyo.SolverFactory("glpk")
        results = solver.solve(instance, tee=False)
        if results.solver.status != pyo.SolverStatus.ok:
            raise RuntimeError(f"Optimization failed at sim={sim_id}, year={y}")

        # After decision, reveal this year's pre-sampled true-world shocks and realized costs.
        year_ar1_real = shocks_ar1.get(str(y), {}).get("1", {})
        year_iid_real = shocks_iid.get(str(y), {}).get("1", {})
        for t in techs_y:
            if stochastic:
                if mr.get(t) is not None:
                    eps_r = float(year_ar1_real.get(t, 0.0))
                    realized_cost[t] = float(
                        np.exp(
                            mr[t] * np.log(max(realized_cost[t], 1e-12))
                            + eps_r
                            + (k[t] if k.get(t) is not None else 0.0)
                        )
                    )
                else:
                    eps_r = float(year_iid_real.get(t, 0.0))
                    real_dist[t] = real_dist.get(t, 0.0) * 0.19 + eps_r

                    z_prev = max(cum_prod[t] - generation_traj[t][-1], 1e-12)
                    z_curr = max(cum_prod[t], z_prev)
                    lr_term_expected = lexp.get(t, 0.0) * np.log(z_curr / z_prev)
                    lr_term_real = lexp_real.get(t, lexp.get(t, 0.0)) * np.log(z_curr / z_prev)

                    c_prev = max(realized_cost[t], 1e-12)
                    c_new = float(np.exp(np.log(c_prev) + lr_term_real + real_dist[t]))
                    realized_cost[t] = c_new

                    # Planner only sees realized costs and cumulative experience;
                    # infer disturbance with expected lexp (no access to true omega).
                    planner_dist[t] = float(np.log(max(c_new, 1e-12) / c_prev) - lr_term_expected)
            else:
                z0 = max(init_cum_prod[t], 1e-12)
                zc = max(cum_prod[t], z0)
                realized_cost[t] = float(init_unit_cost[t] * (zc / z0) ** lexp_real.get(t, lexp.get(t, 0.0)))

        for t in techs_all:
            if t in techs_y:
                cum_gen_traj[t].append(cum_prod[t])
                unit_cost_traj[t].append(realized_cost[t])
                gen_val = float(pyo.value(instance.gen[t.replace(' ', '-')]))
                generation_traj[t].append(gen_val)
                cum_prod[t] += gen_val
                previous_gen[t] = gen_val

                existing_cap = existing_surviving_capacity(vintages, y, t, init_gen)
                new_build = max(0.0, gen_val - existing_cap)
                if new_build > 0:
                    vintages[t].append((y, new_build))
            else:
                unit_cost_traj[t].append(unit_cost_traj[t][-1])
                generation_traj[t].append(0.0)
                cum_gen_traj[t].append(cum_prod[t])

    years_with_2020 = [2020, *years]
    return {
        "sim_id": sim_id,
        "years": years_with_2020,
        "generation_traj": generation_traj,
        "unit_cost_traj": unit_cost_traj,
        "cum_gen_traj": cum_gen_traj,
        # Used by DPS aligned-cost mode: provide realized (sim-specific) learning exponent.
        "learning_exponent": lexp_real,
        # Keep planner expected exponent for diagnostics/reproducibility.
        "learning_exponent_expected": lexp,
        "omega_real": {t: -lexp_real.get(t, 0.0) for t in techs_all},
        "retirement_mode": retirement_mode,
        "default_lifetime": default_lifetime,
        "shared_shocks": {
            "ar1_cost": shocks_ar1,
            "wright_iid": shocks_iid,
        },
    }


def main():
    print(
        f"scenario={scenario}, nsim={nsim}, nscenarios={nscenarios}, "
        f"deterministic={deterministic}, retirement_mode={retirement_mode}, "
        f"forecast={forecast}, risk_averse={risk_averse}, omega={omega}, "
        f"omega_prior_common={omega_prior_common}, risk_quantile={risk_quantile}, "
        f"world_seed={world_seed}, planner_seed={planner_seed}"
    )
    print(f"outdir={outdir}")

    base_technologies = get_base_technologies(scenario)
    (
        init_unit_cost,
        init_cum_prod,
        init_gen,
        lexp,
        sigma,
        mr,
        k,
    ) = build_init_params(base_technologies)
    years, demand = load_demand(scenario, init_gen)
    techs_all = list(base_technologies)
    if SMR_TECH not in techs_all:
        techs_all.append(SMR_TECH)
    if SMR2_TECH not in techs_all:
        techs_all.append(SMR2_TECH)

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    world_samples = pre_sample_real_world_scenarios(
        nsim=nsim,
        years=years,
        techs_all=techs_all,
        lexp=lexp,
        sigma=sigma,
        mr=mr,
        seed_=world_seed,
    )
    world_file = outdir_path / "real_world_cost_scenarios.json"
    with world_file.open("w") as f:
        json.dump(clean({
            "world_seed": world_seed,
            "nsim": nsim,
            "years": years,
            "samples": world_samples,
        }), f, indent=2)
    print(f"Saved fixed real-world scenarios to: {world_file}")

    index = {
        "scenario": scenario,
        "nsim": nsim,
        "nscenarios": nscenarios,
        "seed": seed,
        "retirement_mode": retirement_mode,
        "forecast": forecast,
        "risk_averse": risk_averse,
        "risk_quantile": risk_quantile,
        "omega": omega,
        "omega_prior_common": omega_prior_common,
        "world_seed": world_seed,
        "planner_seed": planner_seed,
        "real_world_scenarios_file": str(world_file),
        "files": [],
    }

    stochastic = not deterministic

    for sim_id in range(1, nsim + 1):
        rng_planner = np.random.default_rng(planner_seed + sim_id)
        sim_output = run_one_sim(
            sim_id=sim_id,
            years=years,
            demand=demand,
            base_technologies=base_technologies,
            init_unit_cost=init_unit_cost,
            init_cum_prod=init_cum_prod,
            init_gen=init_gen,
            lexp=lexp,
            sigma=sigma,
            mr=mr,
            k=k,
            nscenarios=nscenarios,
            stochastic=stochastic,
            rng_planner=rng_planner,
            world_sample=world_samples[str(sim_id)],
        )

        outfile = outdir_path / f"sim_{sim_id:04d}_results_lp_trajectory.json"
        with outfile.open("w") as f:
            json.dump(clean(sim_output), f, indent=2)

        index["files"].append(str(outfile))
        if sim_id % 10 == 0 or sim_id == 1 or sim_id == nsim:
            print(f"Completed sim {sim_id}/{nsim}: {outfile}")

    with (outdir_path / "index.json").open("w") as f:
        json.dump(index, f, indent=2)

    print(f"Saved {nsim} LP trajectories to: {outdir_path}")


if __name__ == "__main__":
    main()
