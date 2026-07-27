import pyomo.environ as pyo
import matplotlib.pyplot as plt
import numpy as np
from annual_LP import model
from energySim._energy_sim_params import scenarios, costparams


# two SMR
# LP determinsitc, update 3/24/2026
# no random at all, just use expected learning rate
# 111_run_annual_lp.py

# create parameters
base_technologies = []
for el in scenarios["fast transition"][0]:
    if (
        el[1] == "electricity" 
        and el[0] not in [
            "EV batteries", 
            "multi-day batteries", 
            "P2Xfuels", 
            "daily batteries"
        ]
    ):
        base_technologies.append(el[0])
for el in scenarios['fast transition'][1]:
    if (
        el == "electricity" 
        and scenarios['fast transition'][1][el] not in [
            "EV batteries", 
            "multi-day batteries", 
            "P2Xfuels", 
            "daily batteries"
        ]
    ):
        base_technologies.append(scenarios['fast transition'][1][el])

# SMR parameters (from costparams); only used from 2030
SMR_TECH = 'SMR electricity'
SMR2_TECH = 'SMR2 electricity'  # 新增技术名
SMR_c0 = costparams['c0'].get(SMR_TECH, 20.0)
SMR_z0 = costparams['z0'].get(SMR_TECH, 1e-9)
SMR_omega = costparams['omega'].get(SMR_TECH, 0.074)
SMR_sigma = costparams['sigma'].get(SMR_TECH, None)
SMR_lexp = -SMR_omega

init_unit_cost = {}
init_cum_prod = {}
init_gen = {}
lexp = {}
sigma = {}
mr = {}
k = {}

# Initialize known generation levels for base technologies
init_gen['coal electricity'] = 35.7
init_gen['gas electricity'] = 22.9
init_gen['nuclear electricity'] = 10.0
init_gen['hydroelectricity'] = 15.2
init_gen['biopower electricity'] = 2.55
init_gen['wind electricity'] = 5.75
init_gen['solar pv electricity'] = 3.0

# Fill parameters for base technologies
for tech in base_technologies:
    init_unit_cost[tech] = costparams['c0'][tech]
    init_cum_prod[tech] = costparams['z0'][tech]
    try:
        lexp[tech] = -costparams['omega'][tech]
    except KeyError:
        lexp[tech] = 0.0
    try:
        sigma[tech] = costparams['sigma'][tech]
    except KeyError:
        sigma[tech] = None
    try:
        mr[tech] = costparams['mr'][tech]
    except KeyError:
        mr[tech] = None
    try:
        k[tech] = costparams['k'][tech]
    except KeyError:
        k[tech] = None

# We will add SMR only from 2030 onward; pre-2030 no entry in sets/params
# Prepare demand and years

# 从 testing_results.py 保存的电力需求JSON读取；JSON覆盖2020-2100
# LP 需求需要 2021-2101，其中：
# - demand_init 取 JSON 的 2020 值
# - 2021-2100 直接用 JSON
# - 2101 用 2100 的值按2%外推
scenario = 'fast transition'
try:
    import json
    with open(f'{scenario}_results_elec_demand.json', 'r') as f:
        elec_json = json.load(f)
    elec_demand_map = elec_json.get('elec_demand', {})
    years = [x for x in range(2021, 2101)]
    demand = {}
    # demand_init 来自 2020
    demand_init = float(elec_demand_map.get('2020'))
    # 填充 2021-2100
    for y in years:
        key = str(y)
        if key in elec_demand_map:
            demand[y] = float(elec_demand_map[key])
        else:
            # 若缺失则用上一年2%增长外推
            prev_val = demand.get(y-1, demand_init)
            demand[y] = prev_val * 1.02
    # 2101 需求（LP需要）
    demand[2101] = float(elec_demand_map.get('2100', demand[2100])) * 1.02
    # 扩展 years 列表以包含 2101
    years.append(2101)
    print('Loaded electricity demand from results_elec_demand.json')
except Exception as e:
    # 回退到旧逻辑：以初始发电量之和为基准并2%增长
    demand_init = sum(init_gen.values())
    years = [x for x in range(2021, 2101)]
    demand = {}
    for y in years:
        demand[y] = demand_init * (1.02)**(y - years[0])
    demand[2101] = demand[2100] * 1.02
    years.append(2101)
    print('Fallback demand construction used:', e)

stochastic = False
nscenarios = 1

# Trajectories and cumulative production for base technologies
cum_prod = {t: init_cum_prod[t] + init_gen.get(t, 0.0) for t in base_technologies}
generation_traj = {t: [init_gen.get(t, 0.0)] for t in base_technologies}
unit_cost_traj = {t: [init_unit_cost[t]] for t in base_technologies}
cum_gen_traj = {t: [init_cum_prod[t]] for t in base_technologies}
unit_cost = {}
dist = {}
prev_dist = {0.0 for t in base_technologies}

# Helper to ensure SMR / SMR2 are initialized when they appear in 2030
def ensure_smr_initialized():
    # SMR1
    if SMR_TECH not in init_unit_cost:
        init_unit_cost[SMR_TECH] = SMR_c0
    if SMR_TECH not in init_cum_prod:
        init_cum_prod[SMR_TECH] = SMR_z0
    if SMR_TECH not in lexp:
        lexp[SMR_TECH] = SMR_lexp
    if SMR_TECH not in sigma:
        sigma[SMR_TECH] = SMR_sigma
    if SMR_TECH not in mr:
        mr[SMR_TECH] = None
    if SMR_TECH not in k:
        k[SMR_TECH] = None
    if SMR_TECH not in init_gen:
        init_gen[SMR_TECH] = 0.0  # SMR has 0 before it starts
    if SMR_TECH not in cum_prod:
        cum_prod[SMR_TECH] = init_cum_prod[SMR_TECH] + init_gen[SMR_TECH]
    if SMR_TECH not in generation_traj:
        generation_traj[SMR_TECH] = [init_gen[SMR_TECH]]
    if SMR_TECH not in unit_cost_traj:
        unit_cost_traj[SMR_TECH] = [init_unit_cost[SMR_TECH]]
    if SMR_TECH not in cum_gen_traj:
        cum_gen_traj[SMR_TECH] = [init_cum_prod[SMR_TECH]]

    # SMR2：全部照抄 SMR 参数
    if SMR2_TECH not in init_unit_cost:
        init_unit_cost[SMR2_TECH] = SMR_c0
    if SMR2_TECH not in init_cum_prod:
        init_cum_prod[SMR2_TECH] = SMR_z0
    if SMR2_TECH not in lexp:
        lexp[SMR2_TECH] = SMR_lexp
    if SMR2_TECH not in sigma:
        sigma[SMR2_TECH] = SMR_sigma
    if SMR2_TECH not in mr:
        mr[SMR2_TECH] = None
    if SMR2_TECH not in k:
        k[SMR2_TECH] = None
    if SMR2_TECH not in init_gen:
        init_gen[SMR2_TECH] = 0.0  # SMR2 has 0 before it starts
    if SMR2_TECH not in cum_prod:
        cum_prod[SMR2_TECH] = init_cum_prod[SMR2_TECH] + init_gen[SMR2_TECH]
    if SMR2_TECH not in generation_traj:
        generation_traj[SMR2_TECH] = [init_gen[SMR2_TECH]]
    if SMR2_TECH not in unit_cost_traj:
        unit_cost_traj[SMR2_TECH] = [init_unit_cost[SMR2_TECH]]
    if SMR2_TECH not in cum_gen_traj:
        cum_gen_traj[SMR2_TECH] = [init_cum_prod[SMR2_TECH]]

# Initialize SMR trajectories so we can pad pre-2030 years
ensure_smr_initialized()

for y in years:
    # Build the set of technologies for this year
    techs_y = list(base_technologies)
    if y >= 2030:
        # Add SMR & SMR2 starting in 2030
        ensure_smr_initialized()
        techs_y.append(SMR_TECH)
        techs_y.append(SMR2_TECH)
        # In 2029 file, we want previous_gen seed for SMR = 0.058; we will write it when y==2030 below

    with open(f"model_params/param_{y}.dat", "w") as f:

        # write sets
        f.write("set TECHNOLOGIES := \n")
        for t in techs_y:
            f.write(t.replace(" ", "-") + "\n")
        f.write(";\n\n")
        f.write("set SCENARIOS := \n")
        for nscen in range(1, nscenarios + 1):
            f.write(f"{str(nscen)}\n")
        f.write(";\n\n")

        # write parameters
        f.write("param unit_cost := \n")
        # Build previous values with safe fallbacks for first appearance (e.g., SMR/SMR2 in 2030)
        prev_unit_cost = {t: unit_cost.get(t, init_unit_cost[t]) for t in techs_y}
        prev_dist = {t: dist.get(t, 0.0) for t in techs_y}
        for nscen in range(1, nscenarios+1):
            for t in techs_y:
                if stochastic:
                    if mr.get(t) is not None:
                        unit_cost[t] = np.exp(
                            mr[t] * np.log(prev_unit_cost[t]) 
                            + np.random.normal(0, sigma.get(t, 0.0)) + (k[t] if k.get(t) is not None else 0.0)
                        )
                    else:
                        dist[t] = prev_dist.get(t, 0.0) * 0.19 + np.random.normal(
                            0, 
                            np.sqrt( (sigma.get(t, 0.0)**2) / (1 + 0.19**2)) if sigma.get(t) is not None else 0.0
                        )
                        unit_cost[t] = np.exp(
                            np.log(prev_unit_cost[t]) + lexp.get(t, 0.0) * np.log(
                                cum_prod[t] / (cum_prod[t] - generation_traj[t][-1])
                            ) + dist[t]
                        )
                else:
                    unit_cost[t] = (
                        init_unit_cost[t] * (cum_prod[t] / init_cum_prod[t]) ** lexp.get(t, 0.0)
                    )
                f.write(f"{t.replace(' ', '-')} {nscen} {str(unit_cost[t])}\n")
        f.write(";\n\n")

        f.write("param previous_gen := \n")
        for t in techs_y:
            # Seed SMR / SMR2 when they first appear (2030)
            if t in (SMR_TECH, SMR2_TECH) and y == 2030:
                value = 0.058
            else:
                value = generation_traj[t][-1] if len(generation_traj[t]) > 0 else init_gen.get(t, 0.0)
            f.write(t.replace(" ", "-") + " " + str(value) + "\n")
        f.write(";\n\n")

        f.write("param demand := \n")
        f.write(str(demand[y])+"\n")
        f.write(";\n\n")

    instance = model.create_instance(f"model_params/param_{y}.dat")

    solver = pyo.SolverFactory('glpk')

    results = solver.solve(instance, tee=False)

    # Check if the solver was successful
    if results.solver.status == pyo.SolverStatus.ok:
        print("Optimization was successful.")
    else:
        print("Optimization failed.")    

    # update cumulative production
    # Iterate over all technologies tracked, padding non-active ones to align series
    for t in generation_traj.keys():
        if t in techs_y:
            # Update with solved values
            cum_gen_traj[t].append(cum_prod[t])
            unit_cost_traj[t].append(unit_cost[t])
            gen_val = pyo.value(instance.gen[t.replace(" ","-")])
            generation_traj[t].append(gen_val)
            cum_prod[t] += gen_val
        else:
            # Tech not active this year: pad series
            last_unit = unit_cost_traj[t][-1] if len(unit_cost_traj[t])>0 else init_unit_cost.get(t, np.nan)
            unit_cost_traj[t].append(last_unit)
            generation_traj[t].append(0.0)
            cum_gen_traj[t].append(cum_prod.get(t, 0.0))


# Visualization: Plot total generation and unit cost over time
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

years = [2020, *years]

all_techs = list(generation_traj.keys())

# Plot Total Generation
for i in all_techs:
    ax1.plot(years, generation_traj[i], label=i)
ax1.set_title('Generation Over Time')
ax1.set_xlabel('Year')
ax1.set_ylabel('Generation')
ax1.legend()

# Plot Unit Cost
for i in all_techs:
    ax2.plot(years, unit_cost_traj[i], label=i)
ax2.set_title('Unit Cost Over Time')
ax2.set_xlabel('Year')
ax2.set_ylabel('Unit Cost')
ax2.legend()

fig, ax = plt.subplots()
for i in all_techs:
    ax.plot(cum_gen_traj[i], unit_cost_traj[i])
ax.set_title('cost dynamics - experience curves')
ax.set_xlabel('Cumulative generation')
ax.set_ylabel('Unit cost')

fig, ax = plt.subplots()
for i in all_techs:
    ax.plot(np.log(cum_gen_traj[i]), np.log(unit_cost_traj[i]))
ax.set_title('cost dynamics - experience curves')
ax.set_xlabel('Cumulative generation')
ax.set_ylabel('Unit cost')

fig, ax = plt.subplots()

checked_techs = []
bottom = [0 for y in years]
for i in all_techs:
    bottom = [sum([generation_traj[ii][y] for ii in checked_techs]) for y in range(len(years))]
    ax.fill_between(
        years, 
        [a+b for a,b in zip(generation_traj[i], bottom)], 
        bottom, 
        label=f'{i}', 
        alpha=0.5
    )
    checked_techs.append(i)

ax.set_xlabel('Year')
ax.set_ylabel('Generation')

plt.legend()


plt.tight_layout()
plt.show()

def print_dominant_tech(target_year):
    if target_year not in years:
        print(f"Dominant tech @{target_year}: year not found")
        return

    idx = years.index(target_year)
    total_generation = sum(float(generation_traj[t][idx]) for t in generation_traj)
    if total_generation <= 0:
        print(f"Dominant tech @{target_year}: none (no generation)")
        return

    shares = {}
    for tech in generation_traj:
        shares[tech] = float(generation_traj[tech][idx]) / total_generation

    dominant = [tech for tech, share in shares.items() if share > 0.5]
    if dominant:
        tech = dominant[0]
        print(f"Dominant tech @{target_year}: {tech} (share={shares[tech]:.4f})")
    else:
        top_tech = max(shares, key=shares.get)
        print(f"Dominant tech @{target_year}: none (top={top_tech}, share={shares[top_tech]:.4f})")

print_dominant_tech(2050)
print_dominant_tech(2070)

# 在保存JSON前，检查并打印SMR在关键年份的发电量
SMR_TECH = 'SMR electricity'
try:
    # years列表以2021开始，前面又加了2020，因此索引=年份-2020
    idx_2050 = 2050 - 2020
    idx_2060 = 2060 - 2020
    idx_2070 = 2070 - 2020
    smr_gen_2050 = generation_traj.get(SMR_TECH, [None]* (idx_2050+1))[idx_2050]
    smr_gen_2060 = generation_traj.get(SMR_TECH, [None]* (idx_2060+1))[idx_2060]
    smr_gen_2070 = generation_traj.get(SMR_TECH, [None]* (idx_2070+1))[idx_2070]
    print(f"SMR generation @2050: {smr_gen_2050}")
    print(f"SMR generation @2060: {smr_gen_2060}")
    print(f"SMR generation @2070: {smr_gen_2070}")
except Exception as e:
    print("SMR generation print failed:", e)

SMR2_TECH = 'SMR2 electricity'
try:
    # years列表以2021开始，前面又加了2020，因此索引=年份-2020
    idx_2050 = 2050 - 2020
    idx_2060 = 2060 - 2020
    idx_2070 = 2070 - 2020
    smr2_gen_2050 = generation_traj.get(SMR2_TECH, [None]* (idx_2050+1))[idx_2050]
    smr2_gen_2060 = generation_traj.get(SMR2_TECH, [None]* (idx_2060+1))[idx_2060]
    smr2_gen_2070 = generation_traj.get(SMR2_TECH, [None]* (idx_2070+1))[idx_2070]
    print(f"SMR2 generation @2050: {smr2_gen_2050}")
    print(f"SMR2 generation @2060: {smr2_gen_2060}")
    print(f"SMR2 generation @2070: {smr2_gen_2070}")
except Exception as e:
    print("SMR2 generation print failed:", e)

# Save LP trajectories for energy sim
import json
output = {
    'years': years,
    'generation_traj': generation_traj,
    'unit_cost_traj': unit_cost_traj,
    'cum_gen_traj': cum_gen_traj
}
with open('111_results_lp_trajectories.json', 'w') as f:
    # Replace numpy types with native
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj
    json.dump(clean(output), f, indent=2)
