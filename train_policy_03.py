# This script trains a new adaptive policy for energy system planning under uncertainty.
# It saves the trained policy (.pth file) to the results folder for later evaluation.
# Usage: Run this script to generate a new policy for a given scenario and label.
#risk averse training



import os
import energySim._energy_sim_model as _energy_sim_model
import energySim._energy_sim_params as _energy_sim_params

label = '020503'
scenario = 'fast transition'
gt_clip = 1
hidden_size = 2
input_norm = False

model = _energy_sim_model.EnergyModel(
    EFgp=_energy_sim_params.scenarios[scenario][0],
    slack=_energy_sim_params.scenarios[scenario][1],
    costparams=_energy_sim_params.costsAssumptions['Way et al. (2022)'],
    gt_clip=gt_clip,
    hidden_size=hidden_size,
    input_norm=input_norm
)

model.mode = 'policy'
model.policy.train(label, iter= 600, batch_size=400, popsize=16, dist=True, agg='percentile', percentile=90) 

os.makedirs('results', exist_ok=True)
policy_path = f'results/{label}_{scenario}_policy.pth'
model.policy.save(policy_path)
print(f"Policy saved to {policy_path}")
