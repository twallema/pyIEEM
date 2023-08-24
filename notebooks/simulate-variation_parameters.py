# Goal of this notebook: simulate BE and SWE model over a period of one year in a purely hypothetical setup (no vacation, no seasonality)
# Use a do-nothing scenario only, assume M_eff is deployed at a certain threshold
# TODO: setup model with equal distribution of sick individuals across spatial patches
# TODO: generate TDPF that triggers general awareness at a certain threshold
# Vary the memory length, speed of leisure and work contact changes
# assess the impact on daily hospitalisations, GDP and employment

import os
import json
import argparse
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

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-ID", "--identifier", help="Calibration identifier")
parser.add_argument("-d", "--date", help="Calibration date")
args = parser.parse_args()

##########################
## change settings here ##
##########################

countries = ['SWE', 'BE']
spatial_units_always = [['Stockholm'], ['Brussels',],]
pars = ['nu',] # 'pi_work', 'pi_leisure', 'mu']
values = [[7, 28, 62], ] # [0.02,], [0.06,], [0.86,]]
states_epi = ['Hin', 'Ih']
states_eco = ['x', 'l']
states = states_epi + states_eco
start_simulation = datetime(2020, 2, 1)
end_simulation = datetime(2020, 5, 1)

######################
## helper functions ##
######################

# helper function to adjust initial condition
def update_initial_condition(spatial_units_always, spatial_units, initial_condition, N):
    """
    A function to divide initial infected over several spatial patches of the model
    """
    # extract contacts
    N_work = N['work']
    N_home = N['home']
    N_other = N['home']
    # pre-allocate output
    output = np.zeros(initial_condition.shape, dtype=float)
    # compute number of contacts per age group per spatial patch (N x G)
    N = np.sum(N_other, axis=0) + np.sum(N_work, axis=0) + np.sum(N_home, axis=0)
    # sum of all infected is always one, only spatial distribution is altered
    infected = 1/len(spatial_units_always)
    # loop over spatial patches you always want to put an infected in
    for j, patchname in enumerate(spatial_units):
        # check if always present
        if patchname in spatial_units_always:
            inf = infected * (N[:, j]/sum(N[:, j]))
            output[:, j] += inf
    return output

#########################
## load model and data ##
#########################

## start loop here
models=[]
inhabitants=[]
print(f"\ninitialising models")
for country,su in zip(countries,spatial_units_always):
    # load samples dictionary
    samples_dict = json.load(open(args.identifier+'_SAMPLES_'+args.date+'.json'))
    # load model SWE with default initial condition
    age_classes = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (
        35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 120)], closed='left')
    model = initialize_epinomic_model(country, age_classes, True, simulation_start=start_simulation, scenarios=f'hypothetical_spatial_spread')
    # set initial infected in Stockholm and Brussels
    init_E = update_initial_condition(su, model.coordinates['spatial_unit'], model.initial_states['E'], model.parameters['N'])
    model.initial_states.update({'E': init_E})
    # load number of inhabitants for normalisation of data
    inhabitant = pd.read_csv(os.path.join(abs_dir, f'../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).sum().values[0]
    # set parameters to calibrated values
    pars = ['nu', 'xi_eff', 'pi_eff', 'pi_work', 'pi_leisure', 'mu', 'amplitude_BE', 'peak_shift_BE', 'amplitude_SWE', 'peak_shift_SWE', 'iota_H', 'iota_F']
    for par in pars:
        model.parameters.update({par: np.mean(samples_dict[par])})
    # eliminate seasonality
    model.parameters.update({
        'amplitude_SWE': 0,
        'amplitude_BE': 0,
    })
    # eliminate other demand shock
    model.parameters.update({
        'shock_exports_goods': 0,
        'shock_exports_services': 0,
        'shock_investment': 0,
    })
    # eliminate labor supply shock
    if country == 'SWE':
        model.parameters.update({'economy_SWE': np.zeros([63,1], dtype=float)})
    elif country == 'BE':
        model.parameters.update({'economy_BE_lockdown_1': np.zeros([63,1], dtype=float),
                                'economy_BE_phaseI': np.zeros([63,1], dtype=float),
                                'economy_BE_phaseII': np.zeros([63,1], dtype=float),
                                'economy_BE_phaseIII': np.zeros([63,1], dtype=float),
                                'economy_BE_phaseIV': np.zeros([63,1], dtype=float),
                                'economy_BE_lockdown_Antwerp': np.zeros([63,1], dtype=float)
        })
    # append to lists
    inhabitants.append(inhabitant)
    models.append(model)

##############
## simulate ##
##############

print(f"starting simulations")

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
    outputs.to_csv(f'simulations-variate_parameters-{par}.csv')

