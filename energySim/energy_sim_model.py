"""

# one SMR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy, copy
import energySim._policy as Policy
import matplotlib.animation as animation
import math
from scipy.optimize import linprog
import json




##### test 
# 1206 06 energy_sim_mode
# LP capacity expansion under uncertainty  
# exogenous capacity exapnsion come from LP solution

# previous update 
# add new technology: SMR! 
# let policy decide electrolyzer, and multi-stage and p2x together to meet VRE
# use Way's stochastic Wright's Law

# model class
class EnergyModel:

    # initialize the model with data
    def __init__(self, EFgp, slack, costparams, mode="exogenous",
                 gt_clip = 0.3, hidden_size = 16, input_norm = False,
                 cost_same_as_lp=False, lp_trajectory_file='1_results_lp_trajectories.json',
                 discount_rate=0.01):

        # growth rates of technologies
        self.EFgp = EFgp

        # max growth rate
        self.gt_clip = gt_clip
        self.discount_rate = float(discount_rate)
        #whether normalize policy inputs
        self.input_norm = input_norm
        
        # initialize technologies, carriers, and sectors
        self.technology = ['oil (direct use)','coal (direct use)',
              'gas (direct use)','coal electricity',
              'gas electricity','nuclear electricity',
              'hydroelectricity','biopower electricity',
              'wind electricity','solar pv electricity','SMR electricity',
              'daily batteries','multi-day storage',
              'electrolyzers','electricity networks', 'P2X']
        
        self.elec_gen_techs = [
            'coal electricity','gas electricity','nuclear electricity',
            'hydroelectricity','biopower electricity',
            'wind electricity','solar pv electricity','SMR electricity'
        ]

        self.carrier = ['oil','coal','gas','electricity','P2Xfuels']
        self.sector = ['transport', 'industry', 'buildings', 'energy']

        # mapping flows from inputs to outputs
        # technology to carriers 
        self.carrierInputs = [[0],[1],[2],[3,4,5,6,7,8,9,10]]
        # carriers to sectors
        self.sectorInputs = [[0,3,4],[0,1,2,3,4],[0,1,2,3,4],[3]]

        
        # define slack variables per sector - no transition below
        self.slack = slack

        # define final year of simulation
        self.yend = 2100

        # initial year
        self.y0 = 2020

        # year counter
        self.y = self.y0

        #self.policy_cache = {}  #cache gt of each tech of last decision year

        # initialize demand dict, 
        # each key is a sector and the item is an array
        self.demand = {}
        for s in self.sector:
            self.demand[s] = np.zeros(self.yend - self.y0 + 1)
        
        # set initial demand values
        self.demand['transport'][0] = 28.6
        self.demand['industry'][0] = 70.6
        self.demand['buildings'][0] = 72.7
        self.demand['energy'][0] = 15.5

        # set demand growth rate 
        self.dgrowth = 0.02

        # set efficiencies from carrier to sector
        self.efficiency = {}
        for s in self.sector:
            for c in self.carrier:
                self.efficiency[c,s] = 0.0
        self.efficiency['oil','transport'] = 0.25
        self.efficiency['electricity','transport'] = 0.8
        self.efficiency['P2Xfuels','transport'] = 0.5
        self.efficiency['oil','industry'] = 0.6
        self.efficiency['coal','industry'] = 0.6
        self.efficiency['gas','industry'] = 0.6
        self.efficiency['electricity','industry'] = 0.8
        self.efficiency['P2Xfuels','industry'] = 0.6
        self.efficiency['oil','buildings'] = 0.7
        self.efficiency['coal','buildings'] = 0.6
        self.efficiency['gas','buildings'] = 0.6
        self.efficiency['electricity','buildings'] = 1.0
        self.efficiency['P2Xfuels','buildings'] = 0.6
        self.efficiency['oil','energy'] = 0.6
        self.efficiency['coal','energy'] = 0.6
        self.efficiency['gas','energy'] = 0.6
        self.efficiency['electricity','energy'] = 1.0

        # create dict for final energy supply in EJ
        # each key is a tuple (carrier, sector) and the item is an array
        self.EF = {}
        for s in self.sector:
            for c in self.carrier:
                self.EF[c,s] = np.zeros(self.yend - self.y0 + 1)

        # initialize final energy supply values
        self.EF['oil','transport'][0] = 110.1
        self.EF['electricity','transport'][0] = 1.34
        self.EF['P2Xfuels','transport'][0] = 2.95e-4
        self.EF['oil','industry'][0] = 12.5
        self.EF['coal','industry'][0] = 33.3
        self.EF['gas','industry'][0] = 27.0
        self.EF['electricity','industry'][0] = 33.6
        self.EF['P2Xfuels','industry'][0] = 2.95e-4
        self.EF['oil','buildings'][0] = 13.8
        self.EF['coal','buildings'][0] = 5.23
        self.EF['gas','buildings'][0] = 29.3
        self.EF['electricity','buildings'][0] = 42.3
        self.EF['P2Xfuels','buildings'][0] = 2.95e-4
        self.EF['oil','energy'][0] = 13.9
        self.EF['coal','energy'][0] = 17.1
        self.EF['gas','energy'][0] = 16.5
        self.EF['electricity','energy'][0] = 15.5

        # useful to account for VRE intermittency
        # self.elec represents the total electricity generation
        self.elec = np.zeros(self.yend - self.y0 + 1)

        # create dict of useful energy supply in EJ
        # each key is a tuple (carrier, sector) and the item is an array
        self.EU = {}
        for s in self.sector:
            for c in self.carrier:
                self.EU[c,s] = np.zeros(self.yend - self.y0 + 1)
                self.EU[c,s][0] = \
                    self.EF[c,s][0] * self.efficiency[c,s]

        # set initial values for useful energy
        self.EU['oil','transport'][0] = 27.5
        self.EU['electricity','transport'][0] = 1.07
        self.EU['P2Xfuels','transport'][0] = 1.48e-4
        self.EU['oil','industry'][0] = 7.5
        self.EU['coal','industry'][0] = 20.0
        self.EU['gas','industry'][0] = 16.2
        self.EU['electricity','industry'][0] = 26.9
        self.EU['P2Xfuels','industry'][0] = 1.77e-4
        self.EU['oil','buildings'][0] = 9.7
        self.EU['coal','buildings'][0] = 3.14
        self.EU['gas','buildings'][0] = 17.6
        self.EU['electricity','buildings'][0] = 42.3
        self.EU['P2Xfuels','buildings'][0] = 1.77e-4
        self.EU['oil','energy'][0] = 8.3
        self.EU['coal','energy'][0] = 10.3
        self.EU['gas','energy'][0] = 9.9
        self.EU['electricity','energy'][0] = 15.5

        # cap on maximum fraction of useful energy 
        # provided by electricty to each sector
        self.xi = {}
        self.xi['electricity','transport'] = 0.8
        self.xi['electricity','industry'] = 0.75
        self.xi['electricity','buildings'] = 0.9

        # efficiency of fossil fuel power plants
        self.zeta = {}
        self.zeta['coal electricity'] = 0.4
        self.zeta['gas electricity'] = 0.5

        # create dict for energy produced by technology
        # each key is a technology and the item is an array
        self.q = {}
        for t in self.technology:
            self.q[t] = np.zeros(self.yend - self.y0 + 1)

        # set initial values for energy produced by technology
        self.q['coal electricity'][0] = 35.7
        self.q['gas electricity'][0] = 22.9
        self.q['nuclear electricity'][0] = 10.0
        self.q['hydroelectricity'][0] = 15.2
        self.q['biopower electricity'][0] = 2.55
        self.q['wind electricity'][0] = 5.75
        self.q['solar pv electricity'][0] = 3.0
        self.q['daily batteries'][0] = 2.23/1000
        self.q['multi-day storage'][0] = 10.8*1e-7
        self.q['electrolyzers'][0] = 0.0001
        self.q['electricity networks'][0] = 0.0001
        self.q['qgrid'] = np.zeros(self.yend - self.y0 + 1)
        self.q['qtransport'] = np.zeros(self.yend - self.y0 + 1)
        self.q['P2X'] = np.zeros(self.yend - self.y0 + 1)
        self.q['P2X'][0] = 3 * 2.95e-4
        self.q['qgrid'][0] = 0.17/1000
        self.q['qtransport'][0] = 2.06/1000
        # idx2030 = 2030 - self.y0
        # self.q['SMR electricity'][idx2030] = 0.058


        # this array is used to represent phase in of P2X fuels
        self.piP2X = np.zeros(self.yend - self.y0 + 1)
        self.piP2X[0] = 1e-10

        # get total electricity generated at time 0
        self.elec[0] = \
            sum([self.EF['electricity',s][0] \
                    for s in self.sector])


        # create dict for cost of energy produced by technology
        # considering when the technology is built
        self.Q = {}
        for t in self.technology:
            self.Q[t] = \
                np.zeros((self.yend - self.y0 + 1, 
                          self.yend - self.y0 + 1))
            
        # set cost assumptions 
        self.costparams = costparams

        # set learning rate technologies
        self.learningRateTechs = self.technology[5:14]

        # initialize cost dictionaries
        self.gridInv = np.zeros(self.yend - self.y0 + 1)
        self.c = {}
        for t in self.technology:
            self.c[t] = np.zeros(self.yend - self.y0 + 1)
        self.C = {}
        for t in self.technology:
            self.C[t] = np.zeros(self.yend - self.y0 + 1)

        #self.ema_electrolyzer = np.log10(max(self.c['electrolyzers'][self.y - self.y0], 1e-9))   # Init ema of electrolyzer unit cost 

        #initialize unit cost and cumulative production dictionaries
        self.u = {}
        self.z = {}
        self.omega = {}

        # for each technology
        for t in self.technology[:14]:
        
            #initialize production and unit cost arrays and 
            self.z[t] = np.zeros(self.yend - self.y0 + 1)
            self.u[t] = np.zeros(self.yend - self.y0 + 1)

            # variables needed for vintaging model
            if t in self.learningRateTechs:
                self.Q[t][0][0] = self.q[t][0]

            # get initial unit cost and cumulative production
            self.c[t][0] = self.costparams['c0'][t]
            try:
                self.z[t][0] = self.costparams['z0'][t]
            except KeyError:
                self.z[t][0] = 0.0
        #self.z['SMR electricity'][idx2030] = 0.058

        self.sample_uncertainties()
        
        # define planning mode
        self.mode = mode
        self.cost_same_as_lp = cost_same_as_lp
        self.lp_trajectory_file = lp_trajectory_file
        self.lp_unit_cost_traj = {}
        self.lp_learning_exponent = {}
        self.lp_shared_shocks = {'ar1_cost': {}, 'wright_iid': {}}
        self.lp_years = []
        self.lp_dist = {}
        # Default to the single realized world stored by new LP nsim runs.
        self.lp_shock_scenario = '1'

        # define policy
        self.policy = Policy.SNES(state_dim=5, 
                                  action_dim=1,
                                  env=self, 
                                  iter=100,
                                  hidden_size= hidden_size)

    def sample_uncertainties(self):

        for t in self.technology[:14]:

            self.omega[t] = 0.0

            # if breakpoints are used, initialize breaks
            if self.costparams['breakpoints']['active'][0]:
                self.breaks = []
                self.lrchanges = []

            if t in self.learningRateTechs:
                self.omega[t] = np.random.normal(\
                    self.costparams['omega'][t], \
                        self.costparams['sigmaOmega'][t])

                # sample first breakpoint
                if self.costparams['breakpoints']['active'][0]:
                    self.breaks.append(\
                        self.z[t][0] * \
                            10**scipy.stats.lognorm.rvs(\
                                self.costparams['breakpoints']\
                                        ['distance - lognormal']
                                        ['shape'], 
                                self.costparams['breakpoints']\
                                        ['distance - lognormal']
                                        ['loc'], 
                                self.costparams['breakpoints']\
                                        ['distance - lognormal']
                                        ['scale'], )
                    )
                    self.lrchanges.append(scipy.stats.norm.rvs(\
                        self.costparams['breakpoints']
                                    ['exponent change - normal']['mu'], \
                            self.costparams['breakpoints']
                                    ['exponent change - normal']['scale']))

    def load_lp_trajectories_from_file(self, filename):
        """Load LP optimization trajectories from JSON file."""
        try:
            #print(f"Attempting to load LP trajectories from: {filename}")
            
            # Check if file exists first
            import os
            if not os.path.exists(filename):
                #print(f"File {filename} does not exist")
                self.lp_trajectories = {}
                return
                
            # Check file size
            file_size = os.path.getsize(filename)
            #print(f"File size: {file_size} bytes")
            
            if file_size == 0:
                #print("File is empty")
                self.lp_trajectories = {}
                return
            
            with open(filename, 'r') as f:
                lp_data = json.load(f)
            
            #print(f"JSON loaded successfully. Keys: {list(lp_data.keys())}")
            
            # Store LP trajectories for use in exogenous mode
            self.lp_trajectories = {}
            
            # Extract generation trajectories from the correct structure
            if 'generation_traj' in lp_data:
                trajectories_data = lp_data['generation_traj']
                #print(f"Found generation_traj with technologies: {list(trajectories_data.keys())}")
            else:
                # Fallback to old structure
                trajectories_data = lp_data
                #print(f"Using direct structure with keys: {list(trajectories_data.keys())}")
            
            # Convert string keys back to technology names and ensure consistent length
            for tech_key, trajectory in trajectories_data.items():
                #print(f"Processing technology: {tech_key}, trajectory length: {len(trajectory) if isinstance(trajectory, list) else 'not a list'}")
                
                # Map technology names (handle both formats)
                tech_mapping = {
                    'coal_electricity': 'coal electricity',
                    'gas_electricity': 'gas electricity', 
                    'nuclear_electricity': 'nuclear electricity',
                    'wind_electricity': 'wind electricity',
                    'solar_pv_electricity': 'solar pv electricity',
                    'SMR_electricity': 'SMR electricity'
                }
                
                # Use mapping if needed, otherwise use the key as is
                tech_name = tech_mapping.get(tech_key, tech_key)
                
                # # Check if trajectory is a list
                # if not isinstance(trajectory, list):
                #     print(f"Warning: trajectory for {tech_key} is not a list: {type(trajectory)}")
                #     continue
                
                # Ensure trajectory has the right length (pad if necessary)
                expected_length = self.yend - self.y0 + 1  # 81 years: 2020-2100
                if len(trajectory) < expected_length:
                    # Pad with last value
                    last_value = trajectory[-1] if trajectory else 0
                    trajectory.extend([last_value] * (expected_length - len(trajectory)))
                
                self.lp_trajectories[tech_name] = trajectory

            # Optional LP cost scenario payload (new format from run_annual_LP.py)
            self.lp_unit_cost_traj = lp_data.get('unit_cost_traj', {}) if isinstance(lp_data, dict) else {}
            self.lp_learning_exponent = lp_data.get('learning_exponent', {}) if isinstance(lp_data, dict) else {}
            self.lp_shared_shocks = lp_data.get(
                'shared_shocks',
                {'ar1_cost': {}, 'wright_iid': {}}
            ) if isinstance(lp_data, dict) else {'ar1_cost': {}, 'wright_iid': {}}
            self.lp_years = lp_data.get('years', []) if isinstance(lp_data, dict) else []
            self.lp_dist = {t: 0.0 for t in self.technology[:14]}
                
            # print(f"Successfully loaded LP trajectories for {len(self.lp_trajectories)} technologies")
            
        except FileNotFoundError:
            print(f"Warning: LP trajectory file '{filename}' not found. Running without LP trajectories.")
            self.lp_trajectories = {}
            self.lp_unit_cost_traj = {}
            self.lp_learning_exponent = {}
            self.lp_shared_shocks = {'ar1_cost': {}, 'wright_iid': {}}
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            self.lp_trajectories = {}
            self.lp_unit_cost_traj = {}
            self.lp_learning_exponent = {}
            self.lp_shared_shocks = {'ar1_cost': {}, 'wright_iid': {}}
        except Exception as e:
            print(f"Error loading LP trajectories: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            self.lp_trajectories = {}
            self.lp_unit_cost_traj = {}
            self.lp_learning_exponent = {}
            self.lp_shared_shocks = {'ar1_cost': {}, 'wright_iid': {}}

    def _use_lp_cost_scenario(self):
        return bool(
            self.cost_same_as_lp and
            isinstance(self.lp_shared_shocks, dict) and
            (self.lp_shared_shocks.get('ar1_cost') or self.lp_shared_shocks.get('wright_iid'))
        )

    def _select_shock_scenario_key(self, year_block):
        if not isinstance(year_block, dict) or not year_block:
            return None
        def _sorted_keys(d):
            keys = list(d.keys())
            try:
                return sorted(keys, key=lambda x: int(x))
            except Exception:
                return sorted(keys)
        if self.lp_shock_scenario == 'last':
            return _sorted_keys(year_block)[-1]
        if str(self.lp_shock_scenario) in year_block:
            return str(self.lp_shock_scenario)
        return _sorted_keys(year_block)[-1]

    def _lp_shock(self, kind, year, tech):
        year_block = self.lp_shared_shocks.get(kind, {}).get(str(year), {})
        scen_key = self._select_shock_scenario_key(year_block)
        if scen_key is None:
            return 0.0
        return float(year_block.get(scen_key, {}).get(tech, 0.0))

    def evaluate_lp_system_cost_from_loaded_traj(self, discount_rate=0.02, discount_end_year=2070):
        if not self.lp_trajectories or not self.lp_unit_cost_traj:
            raise ValueError("LP trajectories or LP unit-cost trajectories are not loaded.")

        years = self.lp_years if self.lp_years else list(range(self.y0, self.yend + 1))
        total = np.zeros(self.yend - self.y0 + 1)
        for t, gen in self.lp_trajectories.items():
            if t not in self.lp_unit_cost_traj:
                continue
            cost = self.lp_unit_cost_traj[t]
            n = min(len(gen), len(cost), len(total))
            total[:n] += np.array(gen[:n]) * np.array(cost[:n]) * 1e9

        discounted = np.zeros_like(total)
        for y in range(self.y0, min(self.yend, discount_end_year) + 1):
            discounted[y - self.y0] = np.exp(-discount_rate * (y - self.y0)) * total[y - self.y0]

        return float(np.sum(discounted))



    ### plotting functions
    def plotDemand(self):
        fig, ax = plt.subplots(figsize=(8,6))
        df = pd.DataFrame(self.demand, 
                          index=range(self.y0,self.yend + 1), 
                          columns=self.demand.keys())
        df.plot.area(stacked=True, lw=0, ax=ax)
        plt.xlim(2018,2075)
        plt.ylabel('EJ')
        plt.xlabel('Year')
        plt.title('Demand')

    def plotFinalEnergyBySource(self,label, filename = None):
        colors = ['black','saddlebrown','darkgray',
                  'saddlebrown','darkgray',
                  'magenta','royalblue',
                  'forestgreen','deepskyblue',
                  'orange','pink','plum','lawngreen', 'burlywood'] 
        df = pd.DataFrame(self.q, 
                          index=range(self.y0,self.yend + 1),
                          columns = self.q.keys())
        cols = df.columns[[not(x) in ['qgrid','qtransport',
                                      'electricity networks',
                                      'electrolyzers'] 
                                      for x in df.columns]]
        df = df[cols]


        # Plot 1: system-wide final energy mix across all technologies.
        # Shows each technology's annual contribution to total final energy supply.
        # annual tech generation
        fig, ax = plt.subplots(figsize=(8,6))
        df.plot.area(stacked=True, lw=0, ax=ax,
                     color=colors, legend=False)
        ax.plot(range(self.y0,self.yend+1), 
                 [sum(
                     [sum(
                         [self.EF[c,s][y - self.y0] 
                          for c in self.carrier]
                         ) for s in self.sector]
                         ) for y in range(self.y0, self.yend+1)],
                           'k--', lw=2)
        ax.set_title('Final energy by source')
        ax.set_xlim(2018,2075)
        ax.set_ylim(0,1500)
        ax.set_ylabel('EJ')
        ax.set_xlabel('Year')

        # Add legend at the bottom
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.15))
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at the bottom for the legend
        if filename:
            plt.savefig(f'results/figures/{label}/{filename}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        return    # Do not create or save the remaining plots.

        # Plot 2: electricity, storage, and P2X output on a logarithmic scale.
        fig, ax = plt.subplots(figsize=(8,6))
        df = df.loc[df.index<self.yend+1]
        df = df[[x for x in df.columns if 'direct use' not in x]]
        df.plot(color=colors[3:], legend=False, ax=ax)
        ax.set_yscale('log', base=10)
        ax.set_ylim(1e-7, 1e3)
        ax.set_ylabel('EJ')
        ax.set_xlabel('Year')
        ax.set_title('Electricity generation, storage'
                     ' capacity and P2X production')
        if filename:
            plt.savefig(f'./figures/{filename}_elec.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

        # Plot 3: annual P2X growth rate.
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(100*(self.q['P2X'][1:]/self.q['P2X'][:-1]-1))
        ax.set_ylim(-20, 120)
        ax.set_ylabel('%')
        ax.set_xlabel('Year')
        ax.set_title('P2X growth rate check')

        # Plot 4: electricity generation by technology.
        fig, ax = plt.subplots(figsize=(8,6))
        df.plot.area(stacked=True, lw=0, ax=ax,
                     color=colors[3:], legend=False)
        ax.plot(\
            range(self.y0+1,self.yend+1), 
            np.sum(np.array(\
                [self.EF['electricity',s][1:self.yend+1] for s in self.sector]
                            ), axis=0), 'k--')
        ax.set_ylim(0, 600)
        ax.set_ylabel('EJ')
        ax.set_xlabel('Year')
        ax.set_title('Electricity generation by primary source')
        if filename:
            plt.savefig(f'./figures/{filename}_primary.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

        # Plot 5: percentage generation mix by technology.
        fig, ax = plt.subplots(figsize=(8,6))
        df['tot'] = df[df.columns[:-1]].sum(axis=1)
        df[df.columns[:-2]] = \
            100*df[df.columns[:-2]].div(df['tot'], axis=0)
        df[df.columns[:-2]].plot.area(\
            stacked=True, lw=0, 
            color=colors[3:], legend=False, ax=ax)
        ax.set_ylim(0, 150)
        ax.set_ylabel('%')
        ax.set_xlabel('Year')
        ax.set_title('Electricity gen. mix by primary source')
        if filename:
            plt.savefig(f'./figures/{filename}_mixprimary.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

        # Plot 6: technology shares highlighting VRE and non-VRE generation.
        fig, ax = plt.subplots(figsize=(8,6))
        df[df.columns[:-2]].plot(color=colors[3:], legend=False, ax=ax)
        df['vre'] = df['solar pv electricity'] + df['wind electricity']
        df['non-vre'] = 100 - df['vre']
        df['non-vre'].plot(color='k', lw=2, ls=':', ax=ax)
        df['vre'].plot(color='k', lw=2, ls='--', ax=ax)
        ax.set_ylim(0, 110)
        ax.set_ylabel('%')
        ax.set_xlabel('Year')
        ax.set_title('Electricity gen. mix by primary source')
        if filename:
            plt.savefig(f'./figures/{filename}_mixprimary.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    def plotIndividualTechAreas(self, filename=None, ncols=4):
        """
        Plot individual area plots for each technology in self.q,
        and combine them into a single large figure.
        """
    

        # Use your color list, cycle if more techs than colors
        colors = ['black','saddlebrown','darkgray',
                'saddlebrown','darkgray',
                'magenta','royalblue',
                'forestgreen','deepskyblue',
                'orange','pink','plum','lawngreen', 'burlywood']

        techs = [t for t in self.q.keys() if t not in [
            'qgrid', 'qtransport', 'electricity networks', 'electrolyzers']]
        ntech = len(techs)
        nrows = math.ceil(ntech / ncols)
        years = range(self.y0, self.yend + 1)

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3*nrows), sharex=True)
        axes = axes.flatten()

        for i, t in enumerate(techs):
            ax = axes[i]
            color = colors[i % len(colors)]
            ax.fill_between(years, self.q[t], color=color, alpha=0.7)
            ax.set_title(t)
            ax.set_xlim(self.y0, self.yend)
            ax.set_ylabel('EJ')
            ax.set_xlabel('Year')
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle('Individual Technology Area Plots', fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if filename:
            plt.savefig(f'./figures/{filename}_individual_areas.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


    def plotCapacityExpansion(self, filename=None, ncols=4):
        """
        Plot capacity expansion over time for all technologies (not just learningRateTechs).
        """
      

        # Include all technologies, not only learning-rate technologies.
        colors = ['black','saddlebrown','darkgray',
              'saddlebrown','darkgray',
              'magenta','royalblue',
              'forestgreen','deepskyblue',
              'orange','pink','plum','lawngreen', 'burlywood']

        techs = [t for t in self.technology if t in self.Q]
        ntech = len(techs)
        nrows = math.ceil(ntech / ncols)
        years = range(self.y0, self.yend + 1)

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3*nrows), sharex=True)
        axes = axes.flatten()

        for i, t in enumerate(techs):
            ax = axes[i]
            # Calculate total operating capacity in each year.
            color = colors[i % len(colors)]
            capacity = [np.sum(self.Q[t][y-self.y0, :]) for y in years]
            ax.plot(years, capacity, color=color, label=f"{t} capacity")
            ax.set_title(t)
            ax.set_ylabel('Capacity (EJ/yr)')
            ax.set_xlabel('Year')
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle('Capacity Expansion Over Time', fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if filename:
            plt.savefig(f'./figures/{filename}_capacity_expansion.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    
    def plotNewBuildsAndRetirements(self, filename=None, ncols=4):
        """
        Plot new builds and retirements for each technology in self.Q.
        Each subplot shows new builds (solid line) and retirements (dashed line) per year.
        Colors match plotFinalEnergyBySource.
        """


        colors = ['black','saddlebrown','darkgray',
                'saddlebrown','darkgray',
                'magenta','royalblue',
                'forestgreen','deepskyblue',
                'orange','pink','plum','lawngreen', 'burlywood']

        techs = [t for t in self.technology if t in self.Q]
        ntech = len(techs)
        nrows = math.ceil(ntech / ncols)
        years = range(self.y0, self.yend + 1)

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3*nrows), sharex=True)
        axes = axes.flatten()

        for i, t in enumerate(techs):
            ax = axes[i]
            color = colors[i % len(colors)]
            # New builds: diagonal of Q
            new_builds = [self.Q[t][y-self.y0, y-self.y0] for y in years]
            # Retirements: capacity built in year y-Lifetime, retired in year y
            L = self.costparams['L'][t] if t in self.costparams['L'] else 0
            retirements = [self.Q[t][y-self.y0-L, y-self.y0-L] if (y-self.y0-L)>=0 else 0 for y in years]
            ax.plot(years, new_builds, color=color, label='New builds', lw=2)
            ax.plot(years, retirements, color=color, ls='--', label='Retirements', lw=2)
            ax.set_title(t)
            ax.set_ylabel('Capacity (EJ/yr)')
            ax.set_xlabel('Year')
            ax.legend()
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle('New Builds and Retirements by Technology', fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if filename:
            plt.savefig(f'./figures/{filename}_newbuilds_retirements.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    
    def get_generation_shares(self):
        #Add a function to get generation share for each tech each year
        """
        Returns a DataFrame: index=year, columns=tech, values=share of each tech in total final energy supplied.
        """
        years = range(self.y0, self.yend + 1)
        techs = [t for t in self.q.keys() if t not in ['qgrid','qtransport','electricity networks','electrolyzers']]
        data = []
        for y in years:
            total = sum([self.q[t][y - self.y0] for t in techs])
            if total > 0:
                shares = [self.q[t][y - self.y0] / total for t in techs]
            else:
                shares = [0 for t in techs]
            data.append(shares)
        df = pd.DataFrame(data, index=years, columns=techs)
        return df

    def plotCostBySource(self):

        colors = ['black','saddlebrown','darkgray',
                  'saddlebrown','darkgray',
                  'magenta','royalblue',
                  'forestgreen','deepskyblue',
                  'orange','pink','plum','lawngreen', 'burlywood']
        
        df = pd.DataFrame(self.C, 
                          index=range(self.y0,self.yend + 1),
                          columns = self.C.keys())
        
        cols = df.columns[[not(x) in ['qgrid','qtransport',
                                      'P2X'] for x in df.columns]]
        df = df[cols]

        # from usd to trillion USD
        df[cols] = df[cols] * 1e-12

        # from billion to trillion
        df['electricity networks'] = self.gridInv * 1e-3 
        
        fig, ax = plt.subplots(figsize=(8,6))
        df.plot.area(stacked=True, lw=0, ax=ax,
                     color=colors, legend=False)
        ax.plot(range(self.y0,self.yend+1), 
                 [sum(
                     [sum(
                         [self.EF[c,s][y - self.y0] 
                          for c in self.carrier]
                         ) for s in self.sector]
                         ) for y in range(self.y0, self.yend+1)], 
                         'k--', lw=2,
                         )
        ax.set_title('Final cost by source')
        ax.set_xlim(2018,2075)
        ax.set_ylim(0,12)
        ax.set_ylabel('Trillion USD')
        ax.set_xlabel('Year')

        df = pd.DataFrame(self.c, index=range(self.y0,self.yend + 1),
                            columns = self.c.keys())
        cols = df.columns[[not(x) in ['qgrid','qtransport',
                                      'P2X'] for x in df.columns]]
        df = df[cols]

        fig, ax = plt.subplots(figsize=(8,6))
        df.plot(color=colors, legend=False, ax=ax)
        ax.set_title('LCOE')
        ax.set_ylabel('USD/GJ')
        ax.set_xlabel('Year')
        ax.set_yscale('log', base=10)

    def plotFinalEnergy(self, label, filename = None):
        df = pd.DataFrame(self.EF, 
                          index=range(self.y0,self.yend + 1), 
                            columns=self.EF.keys())
        df.columns = \
            [str(a[0]+'_'+a[1]) for a in df.columns.to_flat_index()]
        
        sectors = self.sector
        colors=['black','saddlebrown',
                                 'darkgray','lightskyblue','lime']

        legend_labels = ['oil', 'coal', 'gas', 'electricity', 'P2Xfuel']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        for i, s in enumerate(sectors):
            ax = axes[i // 2, i % 2]
            df_ = df[[x for x in df.columns if x.split('_')[1] == s]].copy()
            df_.plot.area(stacked=True, lw=0, color=colors, ax=ax, legend=False)
            ax.set_title(f'Final energy - {s}')
            ax.set_xlim(2018, 2075)
            ax.set_ylim(0, 300)
            ax.set_ylabel('EJ')
            ax.set_xlabel('Year')
            if i == 0:
                handles, _ = ax.get_legend_handles_labels()
        # Place the legend below the full figure.
        if handles:
            fig.legend(handles, legend_labels, loc='lower center', ncol=len(legend_labels), bbox_to_anchor=(0.5, -0.05))
        fig.tight_layout()
        if filename:
            save_dir = f'results/figures/{label}'
            plt.savefig(f'{save_dir}/{filename}_finalenergyBySector.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        
        
        for s in self.sector:
            df_ = df[[x for x in df.columns if x.split('_')[1]==s]].copy()
            df_.plot.area(stacked=True, lw=0, 
                          color=['black','saddlebrown',
                                 'darkgray','lightskyblue','lime'])
            plt.title('Final energy - '+s)
            plt.xlim(2018,2075)
            plt.ylim(0,300)
            plt.ylabel('EJ')
            plt.xlabel('Year')

    def plotS7(self):

        colors = ['black','saddlebrown',
                  'darkgray','lightskyblue','lime']

        fig, ax = plt.subplots(4,4, sharex=True, figsize=(14,10))

        counts = 0
        for s in self.sector:
            countc = 0
            for c in self.carrier:
                ax[counts][0].plot(range(self.y0+1,self.yend+1), 
                            100 * \
                                (self.EU[c,s][1:] / \
                                 (1e-16+self.EU[c,s][:-1])
                                   - 1),
                            color=colors[countc] )
                countc += 1
            counts += 1
        
        [ax[x][0].set_ylim(-20, 80) for x in range(4)]
        [ax[x][0].set_xlim(2018, 2075) for x in range(4)]
        ax[0][0].set_title('Useful energy growth rates [%]')
        ax[0][0].set_ylabel('Transport')
        ax[1][0].set_ylabel('Industry')
        ax[2][0].set_ylabel('Buildings')
        ax[3][0].set_ylabel('Energy sector')


        counts = 0
        for s in self.sector:
            countc = 0
            for c in self.carrier:
                ax[counts][1].plot(range(self.y0+1,2071), 
                                   self.EU[c,s][1:51],
                                    color=colors[countc] )
                countc += 1
            if not s=='energy':
                ax[counts][1].plot(range(self.y0+1,2071), 
                                sum([self.EU[c,s][1:51] 
                                     for c in self.carrier]),
                                'k--' )
            
            counts += 1
        
        [ax[x][1].set_ylim(1e-4, 1e3) for x in range(4)]
        [ax[x][1].set_yscale('log', base=10) for x in range(4)]
        ax[0][1].set_title('Useful energy [EJ]')


        counts = 0
        df = pd.DataFrame(self.EU, 
                          index=range(self.y0, self.yend + 1), 
                columns=self.EU.keys())
        df.columns = \
            [str(a[0]+'_'+a[1]) for a in df.columns.to_flat_index()]
        df = df.loc[(df.index>2020) & (df.index<2071)]
        for s in self.sector:
            df_ = df[[x for x in df.columns if x.split('_')[1]==s]].copy()
            df_.plot.area(stacked=True, lw=0, 
                          color=colors, ax=ax[counts][2],
                          legend=False)
            counts += 1
        [ax[x][2].set_ylim(0, 300) for x in range(4)]
        ax[0][2].set_title('Useful energy [EJ]')

        counts = 0
        df = pd.DataFrame(self.EF, 
                          index=range(self.y0, self.yend + 1), 
                columns=self.EF.keys())
        df.columns = \
            [str(a[0]+'_'+a[1]) for a in df.columns.to_flat_index()]
        df = df.loc[(df.index>2020) & (df.index<2071)]

        for s in self.sector:
            df_ = df[[x for x in df.columns if x.split('_')[1]==s]].copy()
            df_.plot.area(stacked=True, lw=0, 
                          color=colors, ax=ax[counts][3],
                          legend=False)
            counts += 1
        [ax[x][3].set_ylim(0, 300) for x in range(4)]
        ax[0][3].set_title('Final energy [EJ]')

        fig.subplots_adjust(hspace=0.3, wspace=0.3, 
                            top=0.95, bottom=0.075,
                            left=0.07, right=0.95)

    # simulation
    def simulate(self):

        if self.mode == 'exogenous':
            self.load_lp_trajectories_from_file(self.lp_trajectory_file)
        
        self.sample_uncertainties()
        if self._use_lp_cost_scenario():
            self.lp_dist = {t: 0.0 for t in self.technology[:14]}
            for t in self.technology[:14]:
                if t in self.lp_unit_cost_traj and len(self.lp_unit_cost_traj[t]) > 0:
                    self.c[t][0] = float(self.lp_unit_cost_traj[t][0])
        self.y = self.y0
        
        while self.y < self.yend:
            self.step()

        # compute total cost of technologies
        for t in self.technology[:14]:
            for y in range(self.y0, self.yend+1):
                if t not in self.learningRateTechs:
                    self.C[t][y-self.y0] = \
                        self.c[t][y-self.y0] * \
                            1e9 * self.q[t][y-self.y0]
                elif t in ['daily batteries', 'multi-day storage', 'electrolyzers']:
                    cfr = 0.08 * (1 + 0.08)**self.costparams['L'][t] / \
                            ((1 + 0.08)**self.costparams['L'][t] - 1)
                    self.C[t][y-self.y0] = \
                        sum([self.c[t][tau-self.y0] *\
                              1e9 * cfr * \
                                self.Q[t][y-self.y0][tau-self.y0] \
                                    for tau in range(self.y0, y+1)])
                else:
                    self.C[t][y-self.y0] = \
                        sum([self.c[t][tau-self.y0] * \
                             1e9 * self.Q[t][y-self.y0][tau-self.y0] \
                                    for tau in range(self.y0, y+1)])

        # adding grid costs (which are in billion USD)
        self.C[self.technology[14]] = self.gridInv * 1e9                

        # compute total cost
        self.totalCost = np.zeros(self.yend - self.y0 + 1)
        for y in range(self.y0, self.yend+1):
            for t in self.technology[:15]:
                self.totalCost[y-self.y0] += self.C[t][y-self.y0]
            # check if there is any demand deficitf
            total_elec_supply = sum(self.q[t][y-self.y0] for t in self.elec_gen_techs)
            if total_elec_supply < self.elec[y-self.y0] - 1e-1:

                ## using a value of lost load of 10000 USD/MWh 
                ## equivalent to 10000 1e9 USD/ 1e9 MWh
                ## equivalent to 10000 bln USD/PWh
                self.totalCost[y-self.y0] += 10000 * 1/(1000/(60*60)) * \
                    1e9 * max(0, self.elec[y-self.y0] - total_elec_supply)
            

        # compute discounted cost
        # discount rate is configurable (default 1% = 0.01)
        # as in main case of Way et al. (2022)
        self.discountedCost = np.zeros(self.yend - self.y0 + 1)
        for y in range(self.y0, min(self.yend, 2070+1)):
            self.discountedCost[y-self.y0] = \
                np.exp( - self.discount_rate * (y - self.y0) ) * \
                            self.totalCost[y-self.y0]

        self.discountedCost = np.sum(self.discountedCost)

        return self.discountedCost



    # simulation step
    def step(self):

        # this is step 1 in Way et al. (2022) SM page 26
        # 1) Specify the quantity of useful energy 
        # consumed in each sector
        # useful energy demand in EJ - exogenously defined
        # year for which data are given
        for d in self.demand.keys():
            self.demand[d][self.y+1-self.y0] = \
                self.demand[d][self.y-self.y0] * \
                (1 + self.dgrowth)

        # this is step 2 from Way et al. (2022) SM page 26
        # 2 Specify the quantities of energy carriers 
        # used in each sector for each sector
        for s in self.sector:

            # get the defined slack variable
            sl = self.slack[s]

            # for each carrier
            carrs = [self.carrier[x] for x in \
                        self.sectorInputs[self.sector.index(s)]] # Available energy carriers for this sector.

            for c in carrs:
                
                # if carrier not slack 
                if not (c==sl):

                    # get growth rate parameters
                    try:
                        gt0, gT, t1, t2, t3, psi = self.EFgp[c,s] # Growth parameters for carrier c in sector s.
                    # if growth parameters not available, 
                    # consider no useful energy from that carrier
                    except KeyError:
                        self.EU[c,s][self.y+1-self.y0] = 0.0
                        continue

                    # compute growth rate
                    if self.y - self.y0 < t1:
                        gt = 0.01 * gt0
                    elif self.y - self.y0 >= t1 and self.y - self.y0 < t2:
                        s_ = 50 * np.abs(0.01*(gT-gt0)/(t2-t1))
                        gt = 0.01 * gt0 + \
                            0.01 * (gT - gt0) / \
                                (1 + \
                                    np.exp(\
                                        -s_*(self.y - self.y0 -\
                                            t1 - (t2-t1)/2)))
                    else:
                        gt = 0.01 * gT

                    # compute useful energy
                    EUp = self.EU[c,s][self.y-self.y0]
                    if c == 'electricity':
                        maxcap = self.xi[c,s] * \
                            self.demand[s][self.y+1-self.y0]
                    else:
                        maxcap = psi * self.demand[s][self.y+1-self.y0]
                    EUf = min(EUp * (1 + gt), maxcap)  # Grow useful energy by gt without exceeding maxcap.
                    
                    # backcalculate final energy
                    self.EU[c,s][self.y+1-self.y0] = EUf   
                    self.EF[c,s][self.y+1-self.y0] = \
                        self.EU[c,s][self.y+1-self.y0] / self.efficiency[c,s]

            # compute useful energy from slack carrier
            # Apply growth and maximum-share constraints, then recover final energy.
            self.EU[sl,s][self.y+1-self.y0] = \
                max(0, self.demand[s][self.y+1-self.y0] - \
                    sum([self.EU[c,s][self.y+1-self.y0] \
                            for c in carrs if c != sl]))
            self.EF[sl,s][self.y+1-self.y0] = \
                self.EU[sl,s][self.y+1-self.y0] / self.efficiency[sl,s]
        
        # get slack technology for electricity carrier
        sl = self.slack[self.carrier[self.carrier.index('electricity')]]
        # get total electricity generated
        self.elec[self.y+1-self.y0] = \
            sum([self.EF['electricity',s][self.y+1-self.y0] \
                    for s in self.sector])

        # 3 Specify the proportion of 
        # non-dispatchable generation in the electricity mix
        # 4 Increase P2X fuel production 
        # to account for VRE intermittency

        # Increase P2X production as variable renewable generation grows.

        # compute P2X fuel needs
        gt0, gT, t1, t2, t3, psi = \
                    self.EFgp['P2Xfuels','electricity']  
        gt0 = 0
        gT = 20
        t1 = 13
        t2 = t3
        # compute growth rate
        if self.y - self.y0 < t1:
            gt = 0.01 * gt0
        elif self.y - self.y0 >= t1 and self.y - self.y0 < t2:
            s_ = 50 * np.abs(0.01*(gT-gt0)/(t2-t1))
            gt = 0.01 * gt0 +\
                0.01 * (gT - gt0) / \
                    (1+np.exp(\
                        -s_*(self.y - self.y0 - t1 - (t2-t1)/2)))
        else:
            gt = 0.01 * gT
        # adjust P2X fuel production
        self.piP2X[self.y+1-self.y0] = min(5*gt,1)

        # compute P2X fuel production
        self.q['P2X'][self.y+1-self.y0] = \
            sum([self.EF['P2Xfuels',s][self.y+1-self.y0] \
                        for s in self.sector]) + \
            2 * psi * \
                (self.EFgp['solar pv electricity','electricity'][-1] + \
                    self.EFgp['wind electricity','electricity'][-1]) * \
                self.elec[self.y+1-self.y0] * \
                min(self.piP2X[self.y+1-self.y0], 1)
        
        # if self.y == 2099:
        #     print(self.y)
        #     print('STEP4:P2x sector:',  sum([self.EF['P2Xfuels',s][self.y+1-self.y0] \
        #                 for s in self.sector]))
        #     print('STEP4:P2x renewable:',2 * psi * \
        #         (self.EFgp['solar pv electricity','electricity'][-1] + \
        #             self.EFgp['wind electricity','electricity'][-1]) * \
        #         self.elec[self.y+1-self.y0] * \
        #         min(self.piP2X[self.y+1-self.y0], 1) )
         # P2X output equals direct sector demand plus output used to absorb VRE variability.
         # This represents increasing P2X demand as VRE generation grows.

        # 5 Increase total electricity generation if required
        # to account for electrolytic production of P2X fuels.
        self.elec[self.y+1-self.y0] = \
            self.elec[self.y+1-self.y0] - \
                self.EFgp['P2Xfuels','electricity'][-1] * \
                (self.EFgp['solar pv electricity','electricity'][-1] + \
                    self.EFgp['wind electricity','electricity'][-1]) * \
                self.elec[self.y+1-self.y0] * \
                min(self.piP2X[self.y+1-self.y0], 1) + \
            1/0.7 * self.q['P2X'][self.y+1-self.y0]
        

        # 6 Specify the quantity of electricity produced 
        # by each electricity generation technology`

        sl = self.slack['electricity']
   
        # allocate electricity to technologies
        if self.mode == 'exogenous':
            elec_techs = [self.technology[x] for x in self.carrierInputs[self.carrier.index('electricity')]]
            
            year_idx_next = self.y + 1 - self.y0  # write into y+1
            if hasattr(self, 'lp_trajectories') and self.lp_trajectories:
                # Apply LP trajectories (including SMR) if provided
                # if self.y == 2070:
                #     print(f"Year 2070: Using LP trajectories for {len(elec_techs)} technologies")
                #     print(f"Available LP trajectories: {list(self.lp_trajectories.keys())}")
                for t in elec_techs:
                    if t == sl:
                        continue
                    series = self.lp_trajectories.get(t)
                    if series is not None and year_idx_next < len(series):
                        self.q[t][self.y+1-self.y0] = max(0.0, float(series[year_idx_next]))
                    else:
                        # fallback: keep previous value
                        self.q[t][self.y+1-self.y0] = self.q[t][self.y-self.y0]
                    # if self.y == 2040:
                    #     print(f"Technology: {t}")
                    #     print(f"  LP trajectory value for 2041: {series[year_idx_next] if series and year_idx_next < len(series) else 'N/A'}")
                    #     print(f"  Set q[{t}][{self.y+1-self.y0}] = {self.q[t][self.y+1-self.y0]}")
                    #     print(f"  Full q[{t}] array length: {len(self.q[t])}")
                # Slack fills residual
                self.q[sl][self.y+1-self.y0] = max(0, self.elec[self.y+1-self.y0] - \
                    sum([self.q[t][self.y+1-self.y0] for t in elec_techs if t != sl]))
                
                   
            
              

        elif self.mode == 'policy':

            techs = [self.technology[x] for x in self.carrierInputs[self.carrier.index('electricity')]]
            techs.append('electrolyzers')
      

            for t in techs:

                
                # inputs for each technology"
                # 1) unit cost of generation from technology
                # 2) learning rate of technology (actually use cumulative production, could use linear regression to estimate the learning rate)
                # 3) share of generation obtained from technology
                # 4) growth of electricity
                # 5) time

               

                # decide every year 
                year_idx = self.y - self.y0
                 # SMR deployment begins in 2030.
                if t == 'SMR electricity':
                    # The current decision writes to next year's entry: q[t][year_idx + 1].
                    # Decisions from 2020 through 2028 therefore set the next year to zero.
                    if self.y <= 2028:
                        if year_idx + 1 < len(self.q[t]):
                            self.q[t][year_idx + 1] = 0.0
                        continue
                    # In 2029, seed the 2030 value at 0.058 without calling the policy.
                    if self.y == 2029:
                        if year_idx + 1 < len(self.q[t]):
                            self.q[t][year_idx + 1] = 0.058
                            self.z[t][year_idx + 1] = 0.058  
                        continue

                pol_input = [np.log10(self.c[t][self.y-self.y0]),
                                np.log10(self.z[t][self.y-self.y0])/10,
                                (self.y-self.y0)/(self.yend-self.y0),
                                10*((sum(self.q[t_][self.y-self.y0] for t_ in self.elec_gen_techs) / self.elec[self.y-self.y0] - 1)),
                                # 10*((sum([self.q[self.technology[x]][self.y-self.y0] \
                                #     for x in self.carrierInputs[self.carrier.index('electricity')]])/\
                                #     self.elec[self.y-self.y0] - 1)),
                                self.q[t][self.y-self.y0]/self.elec[self.y-self.y0],
                                #np.log10(self.c['electrolyzers'][self.y-self.y0]) # add unit cost of electrolyzer
                                ]

                
                
                
                # Normalize policy inputs.
                if self.input_norm:
                    # Apply z-score normalization only to dimensions 1 and 4.
                    means = np.array([1.06, 0.0, 0.0, 1.09, 0.0])
                    stds  = np.array([0.38, 1.0, 1.0, 0.71, 1.0])
                    
                    arr = np.array(pol_input)
                    arr[0] = (arr[0] - means[0]) / stds[0]
                    arr[3] = (arr[3] - means[3]) / stds[3]
                    pol_input = arr.tolist()
                    
                
                #print('policy inputs:', pol_input)

                ## linear policy
                gt = self.policy.get_action(pol_input)
                # Cap the growth rate.
                gt = min(1.0, gt)

                #make sure gt >=0
                gt = max(0.0, gt) # avoid early retirement

                qp = self.q[t][self.y-self.y0]
                qf = qp * (1 + gt)
                #qf = min(qp * (1 + gt), maxcap)  # if want to add generation cap for tech
                self.q[t][self.y+1-self.y0] = max(0, qf)

        # 7 Calculate the quantities of fossil fuel energy carriers
        #  required by the energy sector

        # get the value of fossil fuel for end use and electricty
        ff_eu_elec2020 = \
            self.EF['oil','transport'][2020-self.y0] + \
            sum([sum([self.EF[c,s][2020-self.y0] \
                        for s in self.sector[1:3]]) \
                        for c in self.carrier[:3]]) + \
            self.q['coal electricity'][2020-self.y0] / \
                self.zeta['coal electricity'] + \
            self.q['gas electricity'][2020-self.y0] / \
                self.zeta['gas electricity'] 


        ff_eu_elec = \
            self.EF['oil','transport'][self.y+1-self.y0] + \
            sum([sum([self.EF[c,s][self.y+1-self.y0] \
                        for s in self.sector[1:3]]) \
                        for c in self.carrier[:3]]) + \
            self.q['coal electricity'][self.y+1-self.y0] / \
                self.zeta['coal electricity'] + \
            self.q['gas electricity'][self.y+1-self.y0] / \
                self.zeta['gas electricity'] 

        # derive energy sector demand for fossil fuels 
        self.EF['oil','energy'][self.y+1-self.y0] = \
            13.9 / (13.9 + 17.1 + 16.5) * \
                ff_eu_elec * (13.9+17.1+16.5) / \
                    ff_eu_elec2020
        self.EF['coal','energy'][self.y+1-self.y0] = \
            17.1 / (13.9 + 17.1 + 16.5) * \
                ff_eu_elec * (13.9+17.1+16.5) / \
                    ff_eu_elec2020
        self.EF['gas','energy'][self.y+1-self.y0] = \
            16.5 / (13.9 + 17.1 + 16.5) * \
                ff_eu_elec * (13.9+17.1+16.5) / \
                    ff_eu_elec2020
        self.EU['oil','energy'][self.y+1-self.y0] = \
            self.EF['oil','energy'][self.y+1-self.y0] * \
                self.efficiency['oil','energy']
        self.EU['gas','energy'][self.y+1-self.y0] = \
            self.EF['gas','energy'][self.y+1-self.y0] * \
                self.efficiency['gas','energy']
        self.EU['coal','energy'][self.y+1-self.y0] = \
            self.EF['coal','energy'][self.y+1-self.y0] * \
                self.efficiency['coal','energy']
        
        # compute direct use of fossil fuels
        self.q['oil (direct use)'][self.y+1-self.y0] = \
            sum([self.EF['oil', s][self.y+1-self.y0] \
                        for s in self.sector])
        self.q['coal (direct use)'][self.y+1-self.y0] = \
            sum([self.EF['coal', s][self.y+1-self.y0] \
                        for s in self.sector])
        self.q['gas (direct use)'][self.y+1-self.y0] = \
            sum([self.EF['gas', s][self.y+1-self.y0] \
                        for s in self.sector])
        
        # 8 Calculate the quantity of daily-cycling batteries required
        # compute short term batteries
        gt0, gT, t1, t2, t3, psi = \
            self.EFgp['daily batteries','electricity']
        self.q['qgrid'][self.y+1 - self.y0] = \
            min((1 + 0.01*gt0) * self.q['qgrid'][self.y-self.y0], 
                psi/365*(self.q['solar pv electricity'][self.y+1-self.y0] + \
                            self.q['wind electricity'][self.y+1-self.y0]))
        gt0, gT, t1, t2, t3, psi = \
            self.EFgp['EV batteries','electricity']
        self.q['qtransport'][self.y+1-self.y0] = \
            min((1 + 0.01*gt0) * self.q['qtransport'][self.y-self.y0], 
                1/365 * \
                    (self.EF['electricity','transport'][self.y+1-self.y0]))
        self.q['daily batteries'][self.y+1-self.y0] = \
            self.q['qgrid'][self.y+1-self.y0] + \
                self.q['qtransport'][self.y+1-self.y0]
        
        if self.mode == "exogenous":

            # 9 Calculate the quantity of 
            # multi-day storage batteries required
            
            # compute long term storage
            gt0, gT, t1, t2, t3, psi = \
                self.EFgp['multi-day batteries','electricity']
            self.q['multi-day storage'][self.y+1-self.y0] = \
                min((1 + 0.01*gt0) * \
                            self.q['multi-day storage'][self.y-self.y0], \
                    psi/365 * (\
                        self.q['solar pv electricity'][self.y+1-self.y0] + \
                        self.q['wind electricity'][self.y+1-self.y0]))
            

        

            # slack 

            """
    
            self.q[sl][self.y+1-self.y0] = \
                max(0, self.elec[self.y+1-self.y0] - \
                    sum([self.q[t][self.y+1-self.y0] for t in \
                        [self.technology[x] for x in \
                            self.carrierInputs\
                                [self.carrier.index('electricity')]] \
                                    if t != sl]))
            """


            # 0822 add step: recompute P2X based on actual solar and wind

            # compute P2X fuel needs
            gt0, gT, t1, t2, t3, psi = \
                        self.EFgp['P2Xfuels','electricity']  
            gt0 = 0
            gT = 20
            t1 = 13
            t2 = t3
            # compute growth rate
            if self.y - self.y0 < t1:
                gt = 0.01 * gt0
            elif self.y - self.y0 >= t1 and self.y - self.y0 < t2:
                s_ = 50 * np.abs(0.01*(gT-gt0)/(t2-t1))
                gt = 0.01 * gt0 +\
                    0.01 * (gT - gt0) / \
                        (1+np.exp(\
                            -s_*(self.y - self.y0 - t1 - (t2-t1)/2)))
            else:
                gt = 0.01 * gT
            # adjust P2X fuel production
            self.piP2X[self.y+1-self.y0] = min(5*gt,1)

            # compute P2X fuel production
            solar_actual = self.q['solar pv electricity'][self.y+1-self.y0]
            wind_actual = self.q['wind electricity'][self.y+1-self.y0]
            vre_total = solar_actual + wind_actual

            self.q['P2X'][self.y+1-self.y0] = \
                sum([self.EF['P2Xfuels',s][self.y+1-self.y0] \
                            for s in self.sector]) + \
                2 * psi * \
                    vre_total * \
                    min(self.piP2X[self.y+1-self.y0], 1)
            # P2X output equals direct sector demand plus output used to absorb VRE variability.
            # This represents increasing P2X demand as VRE generation grows.

            # if self.y == 2099:
            #     #print(self.y)
            #     print('STEP9:P2x sector:',  sum([self.EF['P2Xfuels',s][self.y+1-self.y0] \
            #                 for s in self.sector]))
            #     print('STEP9:P2x reneable:',  2 * psi * \
            #         vre_total * \
            #         min(self.piP2X[self.y+1-self.y0], 1) )



            
            # 10 Calculate the quantity 
            # of electrolyzers required for P2X fuel production
            # compute electrolyzers
            self.q['electrolyzers'][self.y+1-self.y0] = \
                1/(24*365*0.5*0.7) * self.q['P2X'][self.y+1-self.y0]
        
        elif self.mode == 'policy':

            # Calculate the P2X capacity constraint.
            max_p2x_capacity = self.q['electrolyzers'][self.y+1-self.y0] * (24*365*0.5*0.7)
            
            # Satisfy direct sector demand first.
            sector_p2x_demand = sum([self.EF['P2Xfuels',s][self.y+1-self.y0] 
                                    for s in self.sector])
            
            # if self.y == 2040:
            #     print(self.y)
            #     print('from policy p2x:', max_p2x_capacity)
            #     print('p2x sector demand:', sector_p2x_demand)
            
            if max_p2x_capacity >= sector_p2x_demand:
                 # Use spare capacity to absorb VRE generation.
                available_p2x_for_vre = max_p2x_capacity - sector_p2x_demand
            else:
                 # Increase electrolyzer capacity when required to meet sector demand.
                available_p2x_for_vre = 0
                self.q['electrolyzers'][self.y+1-self.y0] = sector_p2x_demand / (24*365*0.5*0.7)

            self.q['P2X'][self.y+1-self.y0] = max(sector_p2x_demand, max_p2x_capacity )
                

            
            vre_total = self.q['solar pv electricity'][self.y+1-self.y0] + \
                        self.q['wind electricity'][self.y+1-self.y0]
             # Total VRE absorption requirement based on the original psi parameter.
            gt0, gT, t1, t2, t3, psi = self.EFgp['P2Xfuels','electricity']
            total_vre_absorption_need = 2 * psi * vre_total * min(self.piP2X[self.y+1-self.y0], 1)

            # VRE absorption assigned to P2X after meeting sector demand.
            p2x_vre_absorption = min(available_p2x_for_vre, total_vre_absorption_need)
            
            # Storage absorbs the remaining VRE generation.
            remaining_vre_for_storage = max(0, total_vre_absorption_need - p2x_vre_absorption)
            
            self.q['multi-day storage'][self.y+1-self.y0] = remaining_vre_for_storage / 365
    
            # if self.y == 2050:
            #     print(self.y)
            #     print('P2x sector:',  sum([self.EF['P2Xfuels',s][self.y+1-self.y0] \
            #                 for s in self.sector]))
            #     print('P2x renewable:', available_p2x_for_vre)
            #     print('multi_day Storage:', self.q['multi-day storage'][self.y+1-self.y0])

        
        #### compute costs

        # grid costs ( in billion USD )
        self.gridInv[self.y-self.y0] = \
            1/3.6*self.costparams['cgrid'] * \
                (self.costparams['elecHist'][self.y-self.y0] - \
            1/2 * (self.elec[self.y-self.y0] - \
                    self.costparams['elecHist'][self.y-self.y0])) + \
            1 / 3.6 * self.costparams['cTripleCap'] * \
                1/2 * (self.elec[self.y-self.y0] - \
                        self.costparams['elecHist'][self.y-self.y0])
        
        # step ahead in vintage model
        for t in self.learningRateTechs:
            for tau in range(self.y0, self.y+1):
                # linear retirement of capacity built prior to 2020
                if tau-self.y0 == 0:
                    self.Q[t][self.y-self.y0][tau-self.y0] = \
                        max(0, 
                            self.q[t][0] * \
                                (1 - (self.y-self.y0)/self.costparams['L'][t]))
                # compute capacity built in year t
                if tau == self.y and self.y-self.y0 > 0:
                    self.Q[t][self.y-self.y0][tau-self.y0] = \
                        max(0, 
                            self.q[t][self.y-self.y0] - \
                                sum([\
                                    self.Q[t][self.y-self.y0][tau_-self.y0] \
                                        for tau_ in range(self.y0,self.y)]))
                # carry over capacity built in year t-1 if still operational
                if tau < self.y and tau - self.y0 > 0 and \
                    self.y - self.y0 > 0:
                    if self.y - self.y0 <= self.costparams['L'][t]:
                        self.Q[t][self.y-self.y0][tau-self.y0] = \
                                    self.Q[t][self.y-self.y0-1][tau-self.y0]
                    else:
                        self.Q[t][self.y-self.y0][tau-self.y0] = \
                            max(0, 
                                self.Q[t][self.y-self.y0-1][tau-self.y0] - \
                                    self.Q[t]\
                                        [self.y-self.y0-self.costparams['L'][t]]\
                                            [tau-self.y0])
                    
            # # normalization over generation produced
            # # spread generation over all all years of installation
            # # used only if generation is decreasing over time
            # self.Q[t][self.y-self.y0] = \
            #     self.Q[t][self.y-self.y0] * \
            #         self.q[t][self.y-self.y0] / \
            #             sum(self.Q[t][self.y-self.y0])

        # compute unit cost of technologies
        for t in self.technology[:14]:

            # update cumulative production
            self.z[t][self.y+1-self.y0] = \
                self.z[t][self.y-self.y0] + \
                    self.q[t][self.y-self.y0]


            # Force LP-aligned cost scenario (for fair LP-vs-DPS comparison)
            if self._use_lp_cost_scenario():
                next_idx = self.y + 1 - self.y0
                year_next = self.y + 1

                # LP mode: directly replay LP unit-cost trajectory
                if self.mode == 'exogenous' and t in self.lp_unit_cost_traj and \
                        next_idx < len(self.lp_unit_cost_traj[t]):
                    self.c[t][next_idx] = float(self.lp_unit_cost_traj[t][next_idx])
                    continue

                # DPS mode: use same shock/learning space as LP, but with DPS deployment
                if t not in self.learningRateTechs:
                    eps = self._lp_shock('ar1_cost', year_next, t)
                    self.c[t][next_idx] = np.exp(
                        self.costparams['mr'][t] * np.log(self.c[t][self.y-self.y0]) +
                        eps + self.costparams['k'][t]
                    )
                else:
                    eps = self._lp_shock('wright_iid', year_next, t)
                    self.lp_dist[t] = self.lp_dist.get(t, 0.0) * 0.19 + eps
                    prev_z = max(self.z[t][self.y-self.y0], 1e-12)
                    curr_z = max(self.z[t][next_idx], prev_z)
                    lexp = self.lp_learning_exponent.get(t, -self.costparams['omega'].get(t, 0.0))
                    self.c[t][next_idx] = np.exp(
                        np.log(self.c[t][self.y-self.y0]) +
                        lexp * np.log(curr_z / prev_z) +
                        self.lp_dist[t]
                    )
                continue

            # if not using experience curves
            # use ar1 model
            if t not in self.learningRateTechs:
                self.c[t][self.y+1-self.y0] = \
                    np.exp(\
                        self.costparams['mr'][t] * \
                            np.log(self.c[t][self.y-self.y0]) + \
                        np.random.normal(0, \
                                        self.costparams['sigma'][t]) + \
                        self.costparams['k'][t])
                
            # if using learning curves, add production to cumulative
            # sample random disturbance and compute unit cost
            # using wright's law with autocorrelated errors
            else:
                self.u[t][self.y+1-self.y0] = \
                    np.random.normal(0, 
                                    np.sqrt(\
                                        self.costparams['sigma'][t]**2 / \
                                                    (1 + 0.19**2) ) )
                
                # is the additional production crossing a breakpoint?
               
                if self.costparams['breakpoints']['active'][0] and\
                    self.z[t][self.y+1-self.y0] > self.breaks[-1]:

                    ## store breakpoints and learning exponent change 
                    breaks = [self.breaks[-1]]
                    lrchanges = [self.lrchanges[-1]]
                    
                    ## sample until next cumulative production is met
                    while self.z[t][self.y+1-self.y0] > self.breaks[-1]:
                        self.breaks.append(\
                            self.breaks[-1] * \
                                10**scipy.stats.lognorm.rvs(\
                                    self.costparams['breakpoints']\
                                            ['distance - lognormal']
                                            ['shape'], 
                                    self.costparams['breakpoints']\
                                            ['distance - lognormal']
                                            ['loc'], 
                                    self.costparams['breakpoints']\
                                            ['distance - lognormal']
                                            ['scale'], )
                            )
                        breaks.append(self.breaks[-1])
                        lrchanges.append(scipy.stats\
                            .norm.rvs(\
                                self.costparams['breakpoints']
                                        ['exponent change - normal']['mu'], \
                                self.costparams['breakpoints']
                                        ['exponent change - normal']['scale']) + \
                            -0.26585740971836214 * self.lrchanges[-1])
                        self.lrchanges.append(lrchanges[-1])
                    
                    ## compute from last production to first breakpoint
                    ## noise is added in this step as it depends on 
                    ## the previous production
                    tempc = np.exp((np.log(self.c[t][self.y-self.y0]) - \
                                    self.omega[t] * \
                                    np.log(breaks[0] / \
                                            self.z[t][self.y-self.y0]) + \
                                    self.u[t][self.y-self.y0] + \
                                        0.19 * self.u[t][self.y-self.y0]))
                    self.omega[t] += lrchanges[0]

                    ## compute from first to last breakpoint
                    for i in range(1,len(breaks)-1):
                        tempc = np.exp((np.log(tempc) - \
                                        self.omega[t] * \
                                        np.log(breaks[i] / \
                                                breaks[i-1])))
                        self.omega[t] += lrchanges[i]
                    
                    ## compute from last breakpoint to current production
                    self.c[t][self.y+1-self.y0] = tempc


                # if no breakpoints are used - standard learning curve
                else:
                    self.c[t][self.y+1-self.y0] = \
                        np.exp((np.log(self.c[t][self.y-self.y0]) - \
                                self.omega[t] * \
                                np.log(self.z[t][self.y+1-self.y0] / \
                                        self.z[t][self.y-self.y0]) + \
                                self.u[t][self.y+1-self.y0] + \
                                    0.19 * self.u[t][self.y-self.y0]))                


        idx2030 = 2030 - self.y0
        self.c['SMR electricity'][idx2030] = self.costparams['c0']['SMR electricity']
        self.z['SMR electricity'][idx2030] = 0.058 
        # Debug: print SMR c and z for 2030-2032.
        # try:
        #     idx2031 = 2031 - self.y0
        #     idx2032 = 2032 - self.y0
        #     idx2033 = 2033-self.y0
        #     print(f"[SMR debug] c(2030)={self.c['SMR electricity'][idx2030]:.6f}, "
        #           f"z(2030)={self.z['SMR electricity'][idx2030]:.6e}")
        #     print(f"[SMR debug] c(2031)={self.c['SMR electricity'][idx2031]:.6f}, "
        #           f"z(2031)={self.z['SMR electricity'][idx2031]:.6e}")
        #     print(f"[SMR debug] c(2032)={self.c['SMR electricity'][idx2032]:.6f}, "
        #           f"z(2032)={self.z['SMR electricity'][idx2032]:.6e}")
        #     print(f"[SMR debug] c(2033)={self.c['SMR electricity'][idx2033]:.6f}, "
        #           f"z(2033)={self.z['SMR electricity'][idx2033]:.6e}")
        # except Exception as e:
        #     print(f"[SMR debug] print failed: {e}")

        
        # advance time counter
        self.y += 1


    def computeCost(self, costparams=None):

        if costparams is not None:
            self.costparams = costparams

        # grid costs ( in billion USD )
        for y in range(self.y0, self.yend+1):
            self.gridInv[y-self.y0] = \
                1/3.6*self.costparams['cgrid'] * \
                    (self.costparams['elecHist'][y-self.y0] - \
                1/2 * (self.elec[y-self.y0] - \
                        self.costparams['elecHist'][y-self.y0])) + \
                1 / 3.6 * self.costparams['cTripleCap'] * \
                    1/2 * (self.elec[y-self.y0] - \
                           self.costparams['elecHist'][y-self.y0])

        # compute vintaging model
        for t in self.learningRateTechs:
            self.Q[t][0][0] = self.q[t][0]
            for y in range(self.y0+1, self.yend):
                for tau in range(self.y0, y+1):
                    if tau-self.y0 == 0:
                        self.Q[t][y-self.y0][tau-self.y0] = \
                            max(0, 
                                self.q[t][0] * \
                                    (1 - (y-self.y0)/self.costparams['L'][t]))
                    if tau == y and y-self.y0 > 0:
                        self.Q[t][y-self.y0][tau-self.y0] = \
                            max(0, 
                                self.q[t][y-self.y0] - \
                                    sum([\
                                        self.Q[t][y-self.y0][tau_-self.y0] \
                                            for tau_ in range(self.y0,y)]))
                    if tau < y and tau - self.y0 > 0 and \
                        y - self.y0 > 0 and \
                            y - self.y0 <= self.costparams['L'][t]:
                        self.Q[t][y-self.y0][tau-self.y0] = \
                                    self.Q[t][y-self.y0-1][tau-self.y0]
                    if tau < y and tau - self.y0 > 0 and \
                        y-self.y0 > self.costparams['L'][t]:
                        self.Q[t][y-self.y0][tau-self.y0] = \
                            max(0, 
                                self.Q[t][y-self.y0-1][tau-self.y0] - \
                                    self.Q[t]\
                                        [y-self.y0-self.costparams['L'][t]]\
                                            [tau-self.y0])
            
            # normalization over generation produced
            # assume generation is equally spread 
            # over the different years of installation
            # (needed if generation is decreasing over time, rarely)
            if t in self.learningRateTechs:
                for y in range(self.y0+1, self.yend):
                    self.Q[t][y-self.y0] = \
                        self.Q[t][y-self.y0] * \
                            self.q[t][y-self.y0] / \
                                sum(self.Q[t][y-self.y0])
        
        # compute unit cost of technologies

        # for each technology
        for t in self.technology[:14]:
                
            # iterate over the years
            for y in range(self.y0, self.yend):

                # if year 0, initialize unit cost
                if y == self.y0:
                    self.c[t][0] = self.costparams['c0'][t]

                    # if experience curves are used
                    # get initial cumulative production
                    # and sample uncertain parameters
                    if t in self.learningRateTechs:
                        self.z[t][0] = self.costparams['z0'][t]
                        self.omega[t] = np.random.normal(\
                            self.costparams['omega'][t], \
                                self.costparams['sigmaOmega'][t])

                        # sample first breakpoint
                        if self.costparams['breakpoints']['active'][0]:
                            self.breaks.append(\
                                self.z[t][0] * \
                                    10**scipy.stats.lognorm.rvs(\
                                        self.costparams['breakpoints']\
                                                ['distance - lognormal']
                                                ['shape'], 
                                        self.costparams['breakpoints']\
                                                ['distance - lognormal']
                                                ['loc'], 
                                        self.costparams['breakpoints']\
                                                ['distance - lognormal']
                                                ['scale'], )
                            )
                            self.lrchanges.append(scipy.stats.norm.rvs(\
                                self.costparams['breakpoints']
                                            ['exponent change - normal']['mu'], \
                                    self.costparams['breakpoints']
                                            ['exponent change - normal']['scale']))
                
                # if not using experience curves
                # use ar1 model
                if t not in self.learningRateTechs:
                    self.c[t][y+1-self.y0] = \
                        np.exp(\
                            self.costparams['mr'][t] * \
                                np.log(self.c[t][y-self.y0]) + \
                            np.random.normal(0, \
                                            self.costparams['sigma'][t]) + \
                            self.costparams['k'][t])
                
                # if using learning curves, add production to cumulative
                # sample random disturbance and compute unit cost
                # using wright's law with autocorrelated errors
                else:
                    self.u[t][y+1-self.y0] = \
                        np.random.normal(0, 
                                        np.sqrt(\
                                            self.costparams['sigma'][t]**2 / \
                                                        (1 + 0.19**2) ) )
                
                    # is the additional production crossing a breakpoint?
                    if self.costparams['breakpoints']['active'][0] and\
                        self.z[t][y+1-self.y0] > self.breaks[-1]:

                        ## store breakpoints and learning exponent change 
                        breaks = [self.breaks[-1]]
                        lrchanges = [self.lrchanges[-1]]
                        
                        ## sample until next cumulative production is met
                        while self.z[t][y+1-self.y0] > self.breaks[-1]:
                            self.breaks.append(\
                                self.breaks[-1] * \
                                    10**scipy.stats.lognorm.rvs(\
                                        self.costparams['breakpoints']\
                                                ['distance - lognormal']
                                                ['shape'], 
                                        self.costparams['breakpoints']\
                                                ['distance - lognormal']
                                                ['loc'], 
                                        self.costparams['breakpoints']\
                                                ['distance - lognormal']
                                                ['scale'], )
                                )
                            breaks.append(self.breaks[-1])
                            lrchanges.append(scipy.stats\
                                .norm.rvs(\
                                    self.costparams['breakpoints']
                                            ['exponent change - normal']['mu'], \
                                    self.costparams['breakpoints']
                                            ['exponent change - normal']['scale']) + \
                                -0.26585740971836214 * self.lrchanges[-1])
                            self.lrchanges.append(lrchanges[-1])
                        
                        ## compute from last production to first breakpoint
                        ## noise is added in this step as it depends on 
                        ## the previous production
                        tempc = np.exp((np.log(self.c[t][y-self.y0]) - \
                                        self.omega[t] * \
                                        np.log(breaks[0] / \
                                                self.z[t][y-self.y0]) + \
                                        self.u[t][y-self.y0] + \
                                            0.19 * self.u[t][y-self.y0]))
                        self.omega[t] += lrchanges[0]

                        ## compute from first to last breakpoint
                        for i in range(1,len(breaks)-1):
                            tempc = np.exp((np.log(tempc) - \
                                            self.omega[t] * \
                                            np.log(breaks[i] / \
                                                    breaks[i-1])))
                            self.omega[t] += lrchanges[i]
                        
                        ## compute from last breakpoint to current production
                        self.c[t][y+1-self.y0] = tempc


                    # if no breakpoints are used - standard learning curve
                    else:
                        self.c[t][y+1-self.y0] = \
                            np.exp((np.log(self.c[t][y-self.y0]) - \
                                    self.omega[t] * \
                                    np.log(self.z[t][y+1-self.y0] / \
                                            self.z[t][y-self.y0]) + \
                                    self.u[t][y+1-self.y0] + \
                                        0.19 * self.u[t][y-self.y0]))


        # compute total cost of technologies
        for t in self.technology[:14]:
            for y in range(self.y0, self.yend+1):
                if t not in self.learningRateTechs:
                    self.C[t][y-self.y0] = \
                        self.c[t][y-self.y0] * \
                            1e9 * self.q[t][y-self.y0]
                elif t in ['daily batteries', 'multi-day storage', 'electrolyzers']:
                    cfr = 0.08 * (1 + 0.08)**self.costparams['L'][t] / \
                            ((1 + 0.08)**self.costparams['L'][t] - 1)
                    self.C[t][y-self.y0] = \
                        sum([self.c[t][tau-self.y0] *\
                              1e9 * cfr * \
                                self.Q[t][y-self.y0][tau-self.y0] \
                                    for tau in range(self.y0, y+1)])
                else:
                    self.C[t][y-self.y0] = \
                        sum([self.c[t][tau-self.y0] * \
                             1e9 * self.Q[t][y-self.y0][tau-self.y0] \
                                    for tau in range(self.y0, y+1)])

        # adding grid costs (which are in billion USD)
        self.C[self.technology[14]] = self.gridInv * 1e9

        # compute total cost
        self.totalCost = np.zeros(self.yend - self.y0 + 1)
        for y in range(self.y0, self.yend+1):
            for t in self.technology[:15]:
                self.totalCost[y-self.y0] += self.C[t][y-self.y0]

        # compute discounted cost
        # discount rate is configurable (default 1% = 0.01)
        # as in main case of Way et al. (2022)
        self.discountedCost = np.zeros(self.yend - self.y0 + 1)
        for y in range(self.y0, min(self.yend, 2070+1)):
            self.discountedCost[y-self.y0] = \
                np.exp( - self.discount_rate * (y - self.y0) ) * \
                            self.totalCost[y-self.y0]

        self.discountedCost = np.sum(self.discountedCost)

        return self.totalCost, self.discountedCost
    
    # create a copy of the model
    def copy(self):
        return copy.deepcopy(self)
    
    # make a gif reporting cost trajectories and electricity generation
    # for all technologies updating lines over time
    def make_gif(self, filename):
        fig, ax = plt.subplots(1, 2, figsize=(12, 7))

        colors = ['black','saddlebrown','darkgray',
            'saddlebrown','darkgray',
            'magenta','royalblue',
            'forestgreen','deepskyblue',
            'orange','pink','plum','lawngreen', 'burlywood'] 
        
        colors = colors[3:10]
        
        costs = []
        cprod = []
        gen = []
        demand = self.elec
        for t in self.technology[3:10]:
            costs.append(self.c[t])
            cprod.append(self.z[t])
            gen.append(self.q[t])
        
        costs = np.array(costs)
        cprod = np.array(cprod)
        gen = np.array(gen)

        for i in range(costs.shape[0]):
            ax[0].plot(cprod[i,:], costs[i,:], color=colors[i])
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[1].stackplot([x for x in range(self.y0, self.yend+1)], 
                                gen, colors=colors,
                                labels=self.technology[3:10])
        ax[1].plot([x for x in range(self.y0, self.yend+1)], 
                   demand, color='k', ls='--', label='Demand')
        ax[0].set_xlabel('Cumulative production (EJ)')
        ax[0].set_ylabel('Unit cost (USD/GJ)')
        ax[1].set_xlabel('Years')
        ax[1].set_ylabel('Electricity generation (EJ)')

        handles, labels = ax[1].get_legend_handles_labels()

        fig.legend(handles, labels, loc='lower center', ncol=4)
        
        plt.subplots_adjust(top=0.95, bottom=0.25, 
                            left=0.1, right=0.95,
                              hspace=0.25, wspace=0.35)

        for l in ax[0].get_lines():
            l.remove()
        for c in ax[1].collections:
            c.remove()
        for l in ax[1].get_lines():
            l.remove()

        plt.pause(0.01)


        ani = animation.FuncAnimation(fig, self.update_gif, 
                                         fargs=[[costs, cprod, 
                                                gen, demand, 
                                                fig, colors]],
                                         repeat=True,
                                         frames=[x for x \
                                                 in range(self.y0, self.yend+1)], 
                                         interval=100)    
        

        # save animation as a gif
        writer = animation.PillowWriter(fps=5,
                    metadata=dict(artist='Me'),)
        ani.save('./figures/trajs_decisions'+filename+'.gif', writer=writer)

    def update_gif(self, yy, fargs):

        costs, cprod, gen, demand, fig, colors = fargs

        costs = costs[:,:yy-self.y0]
        cprod = cprod[:,:yy-self.y0]
        gen = gen[:,:yy-self.y0]
        demand = demand[:yy-self.y0]

        ax = fig.get_axes()

        for l in ax[0].get_lines():
            l.remove()
        for c in ax[1].collections:
            c.remove()
        for l in ax[1].get_lines():
            l.remove()

        for i in range(costs.shape[0]):
            ax[0].plot(cprod[i], costs[i], color=colors[i])

        ax[1].stackplot([x for x in range(self.y0, yy)], gen, colors=colors)
        ax[1].plot([x for x in range(self.y0, yy)],
                   demand, color='k', ls='--')
