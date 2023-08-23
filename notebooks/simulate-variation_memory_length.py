# Goal of this notebook: simulate BE and SWE model over a period of one year in a purely hypothetical setup (no vacation, no seasonality)
# Use a do-nothing scenario only, assume M_eff is deployed at a certain threshold
# TODO: setup model with equal distribution of sick individuals across spatial patches
# TODO: generate TDPF that triggers general awareness at a certain threshold
# Vary the memory length, speed of leisure and work contact changes
# assess the impact on daily hospitalisations, GDP and employment

import os
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
from pyIEEM.models.utils import initialize_epinomic_model

# Hope to disable threading thing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# define all paths absolute
abs_dir = os.path.dirname(__file__)

##########################
## change settings here ##
##########################

pars = ['nu',] # 'pi_work', 'pi_leisure', 'mu']
values = [[7, 28, 62], ] # [0.02,], [0.06,], [0.86,]]
states_epi = ['Hin', 'Ih']
states_eco = ['x', 'l']
states = states_epi + states_eco
start_simulation = datetime(2020, 2, 1)
end_simulation = datetime(2020, 2, 7)

######################
## helper functions ##
######################

# helper function to adjust initial condition
def update_initial_condition(n, initial_condition, N_other, N_work):
    """
    distribute `n` number of infected over the model's age groups according to the amount of social contact
    """
    assert len(n) == initial_condition.shape[1]
    # compute number of contacts per age group per spatial patch (N x G)
    N = np.sum(N_other, axis=0) + np.sum(N_work, axis=0)
    # pre-allocate output
    output = np.zeros(initial_condition.shape, dtype=float)
    # loop over spatial patches
    for j, inf in enumerate(n):
        # distribute over age groups
        inf = inf * (N[:, 0]/sum(N[:, 0]))
        # determine index of age group to drop infected in
        # i = np.random.choice(len(N[:, 0]), p=N[:, 0]/sum(N[:, 0]))
        # assign to output
        output[:, j] = inf
    return output

#################
## load models ##
#################

# use default age classes
age_classes = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (
    35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 120)], closed='left')
# load model and update intial condition
countries = ['BE', 'SWE']
models = []
inhabitants = []
for country in countries:
    # load total number of inhabitants
    inhabitants.append(pd.read_csv(os.path.join(
        abs_dir, f'../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).sum().values[0])
    # load model
    model = initialize_epinomic_model(country, age_classes, True, start_simulation, scenarios='hypothetical_pure')
    # update initial condition: one infected per spatial patch, distributed over all age groups
    model.initial_states['E'] = update_initial_condition(np.ones(model.parameters['G'].shape[0]), model.initial_states['E'],
                                                            model.parameters['N']['other'], model.parameters['N']['work'])
    # update parameters with calibrated values
    model.parameters.update({
        'nu': 20,
        'xi_eff': 0.40,
        'pi_eff': 0.06,
        'pi_work': 0.02,
        'pi_leisure': 0.06,
        'mu': 0.86,
        'iota_H': 7.1,
        'iota_F': 6.6,
    })
    # disable seasonality
    model.parameters.update({'amplitude_BE': 0, 'amplitude_SWE': 0})
    # append to output
    models.append(model)

##############
## simulate ##
##############

# simulation loop
copypar = models[0].parameters.copy()
for i, (par,vals) in enumerate(zip(pars,values)):
    print(f"working on parameter '{par}'")
    # pre-made dataframe with desired formatting
    lvl = [countries,] + [vals,] + [pd.date_range(start=start_simulation, end=end_simulation, freq='D'), ]
    nm = ['country',] + [par,] + ['date',]
    index = pd.MultiIndex.from_product(lvl, names=nm)
    columns = states        
    outputs = pd.DataFrame(0, index=index, columns=columns, dtype=float)                           
    # loop over values
    for v in vals:
        print(f"\t value: {v}")
        for country, model, inhabitant in zip(countries, models, inhabitants):
            # set parameter
            model.parameters.update({par: v})
            # simulate
            simout = model.sim([start_simulation, end_simulation])
            # aggregate and send to output
            for j, state in enumerate(states):
                # aggregate over all dimensions except time
                out_copy = simout[state]
                for dimension in simout[state].dims:
                    if dimension != 'date':
                        out_copy = out_copy.sum(dim=dimension)
                # normalise with demographics if epi state
                if state in states_epi:
                    out_copy /= inhabitant/100000
                # normalise with initial amount if epi state 
                elif state in states_eco:
                    out_copy /= out_copy.isel(date=0).values
                # add to output
                outputs.loc[(country,) + (v,) + (slice(None),), states[j]] = out_copy.values
            # reset parameters
            model.parameters.update({par: copypar[par]})
    # save output
    outputs.to_csv(f'simulations_hypothetical_pure_{par}.csv')

