"""
File: run_annual_LP.py
Last Updated: 2026-02-25
Purpose:
- Run annual LP capacity-expansion rollout for `nsim` stochastic simulations.
- Save one per-simulation JSON with LP generation/cost trajectories and
  shared stochastic shocks for aligned LP-vs-DPS evaluation.

Inputs:
- Scenario demand file: `<scenario>_results_elec_demand.json` (if available).
- LP model (`annual_LP.py`) and energy cost parameters.

Outputs:
- `results/lp_nsim_runs/sim_XXXX_results_lp_trajectory.json`
- `results/lp_nsim_runs/index.json`
"""

import json
from pathlib import Path

import numpy as np
import pyomo.environ as pyo

from annual_LP import model
from energySim._energy_sim_params import scenarios, costparams


SMR_TECH = "SMR electricity"

# ===================== USER CONFIG =====================
nsim = 200
scenario = "fast transition"
nscenarios = 100 # every year lp optmize, use nsceanrio cost as expected objective 
seed = 0
outdir = "results/lp_nsim_runs"
deterministic = False
# ======================================================


def get_base_technologies(transition_name: str):
    base_technologies = []
    for el in scenarios[transition_name][0]:
        if (
            el[1] == "electricity"
            and el[0]
            not in [
                "EV batteries",
                "multi-day batteries",
                "P2Xfuels",
                "daily batteries",
            ]
        ):
            base_technologies.append(el[0])
    for el in scenarios[transition_name][1]:
        if (
            el == "electricity"
            and scenarios[transition_name][1][el]
            not in [
                "EV batteries",
                "multi-day batteries",
                "P2Xfuels",
                "daily batteries",
            ]
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

    # Initialize SMR lazily from 2030 onward.
    init_unit_cost.setdefault(SMR_TECH, costparams["c0"].get(SMR_TECH, 20.0))
    init_cum_prod.setdefault(SMR_TECH, costparams["z0"].get(SMR_TECH, 1e-9))
    lexp.setdefault(SMR_TECH, -costparams["omega"].get(SMR_TECH, 0.074))
    sigma.setdefault(SMR_TECH, costparams["sigma"].get(SMR_TECH))
    mr.setdefault(SMR_TECH, costparams["mr"].get(SMR_TECH))
    k.setdefault(SMR_TECH, costparams["k"].get(SMR_TECH))
    init_gen.setdefault(SMR_TECH, 0.0)

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
    rng,
):
    techs_all = list(base_technologies)
    if SMR_TECH not in techs_all:
        techs_all.append(SMR_TECH)

    cum_prod = {t: init_cum_prod[t] + init_gen.get(t, 0.0) for t in techs_all}
    generation_traj = {t: [init_gen.get(t, 0.0)] for t in techs_all}
    unit_cost_traj = {t: [init_unit_cost[t]] for t in techs_all}
    cum_gen_traj = {t: [init_cum_prod[t]] for t in techs_all}

    # Shared shocks for fair LP-vs-DPS comparison.
    shocks_ar1 = {str(y): {} for y in years}
    shocks_iid = {str(y): {} for y in years}

    unit_cost = {}
    dist = {}

    for y in years:
        techs_y = list(base_technologies)
        if y >= 2030:
            techs_y.append(SMR_TECH)

        param_file = Path("model_params") / f"param_sim{sim_id:04d}_{y}.dat"
        with param_file.open("w") as f:
            f.write("set TECHNOLOGIES := \n")
            for t in techs_y:
                f.write(f"{t.replace(' ', '-')}\n")
            f.write(";\n\n")

            f.write("set SCENARIOS := \n")
            for nscen in range(1, nscenarios + 1):
                f.write(f"{nscen}\n")
            f.write(";\n\n")

            f.write("param unit_cost := \n")
            prev_unit_cost = {t: unit_cost.get(t, init_unit_cost[t]) for t in techs_y}
            prev_dist = {t: dist.get(t, 0.0) for t in techs_y}
            year_shocks_ar1 = {}
            year_shocks_iid = {}

            for nscen in range(1, nscenarios + 1):
                scen_key = str(nscen)
                year_shocks_ar1.setdefault(scen_key, {})
                year_shocks_iid.setdefault(scen_key, {})
                for t in techs_y:
                    if stochastic:
                        if mr.get(t) is not None:
                            eps = float(rng.normal(0.0, sigma.get(t, 0.0) or 0.0))
                            year_shocks_ar1[scen_key][t] = eps
                            unit_cost[t] = np.exp(
                                mr[t] * np.log(prev_unit_cost[t])
                                + eps
                                + (k[t] if k.get(t) is not None else 0.0)
                            )
                        else:
                            sd = (
                                np.sqrt((sigma.get(t, 0.0) ** 2) / (1 + 0.19**2))
                                if sigma.get(t) is not None
                                else 0.0
                            )
                            eps = float(rng.normal(0.0, sd))
                            year_shocks_iid[scen_key][t] = eps
                            dist[t] = prev_dist.get(t, 0.0) * 0.19 + eps
                            unit_cost[t] = np.exp(
                                np.log(prev_unit_cost[t])
                                + lexp.get(t, 0.0)
                                * np.log(cum_prod[t] / (cum_prod[t] - generation_traj[t][-1]))
                                + dist[t]
                            )
                    else:
                        unit_cost[t] = init_unit_cost[t] * (
                            cum_prod[t] / init_cum_prod[t]
                        ) ** lexp.get(t, 0.0)

                    f.write(f"{t.replace(' ', '-')} {nscen} {unit_cost[t]}\n")
            f.write(";\n\n")

            shocks_ar1[str(y)] = year_shocks_ar1
            shocks_iid[str(y)] = year_shocks_iid

            f.write("param previous_gen := \n")
            for t in techs_y:
                if t == SMR_TECH and y == 2030:
                    value = 0.058
                else:
                    value = generation_traj[t][-1]
                f.write(f"{t.replace(' ', '-')} {value}\n")
            f.write(";\n\n")

            f.write("param demand := \n")
            f.write(f"{demand[y]}\n")
            f.write(";\n\n")

        instance = model.create_instance(str(param_file))
        solver = pyo.SolverFactory("glpk")
        results = solver.solve(instance, tee=False)
        if results.solver.status != pyo.SolverStatus.ok:
            raise RuntimeError(f"Optimization failed at sim={sim_id}, year={y}")

        for t in techs_all:
            if t in techs_y:
                cum_gen_traj[t].append(cum_prod[t])
                unit_cost_traj[t].append(unit_cost[t])
                gen_val = float(pyo.value(instance.gen[t.replace(' ', '-')]))
                generation_traj[t].append(gen_val)
                cum_prod[t] += gen_val
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
        "learning_exponent": lexp,
        "shared_shocks": {
            "ar1_cost": shocks_ar1,
            "wright_iid": shocks_iid,
        },
    }


def main():
    print(f"scenario={scenario}, nsim={nsim}, nscenarios={nscenarios}, deterministic={deterministic}")
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

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    index = {
        "scenario": scenario,
        "nsim": nsim,
        "nscenarios": nscenarios,
        "seed": seed,
        "files": [],
    }

    stochastic = not deterministic

    for sim_id in range(1, nsim + 1):
        rng = np.random.default_rng(seed + sim_id)
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
            rng=rng,
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
