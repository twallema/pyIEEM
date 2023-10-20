import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
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

# measures
multiplier = [0.94, 1.01] # ad-hoc to reach nominal IC capacity
triggers = [70*(1000/600), 70] # Hin in BE on 2020-03-15 is equal to 70 days
triggers = np.array(triggers)*np.array(multiplier)
length_measures_list = [2*28, 3*28, 4*28, 5*28]
policies_df = pd.read_csv(os.path.join(abs_dir, f'../data/interim/eco/policies/policies_BE.csv'), index_col=[0], header=[0])
economic_closures = np.expand_dims(policies_df['lockdown_1'].values, axis=1)
telework = 1
social_restrictions = 1 
# others
countries = ['BE', 'SWE']
spatial_units_always = [['Brussels',],['Stockholm',]]
states_epi = ['Hin', 'Ih']
states_eco = ['x', 'l']
states = states_epi + states_eco
start_simulation = datetime(2020, 2, 1)
end_simulation = datetime(2021, 3, 1)

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
    model = initialize_epinomic_model(country, age_classes, True, simulation_start=start_simulation, scenarios=f'prevention_paradox')
    # set initial infected in Stockholm and Brussels
    init_E = update_initial_condition(su, model.coordinates['spatial_unit'], model.initial_states['E'], model.parameters['N'])
    model.initial_states.update({'E': init_E})
    # load number of inhabitants for normalisation of data
    inhabitant = pd.read_csv(os.path.join(abs_dir, f'../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).sum().values[0]
    # set parameters to calibrated values
    pars = ['nu', 'xi_eff', 'pi_eff', 'pi_work', 'pi_leisure', 'mu', 'amplitude_BE', 'peak_shift_BE', 'amplitude_SWE', 'peak_shift_SWE', 'iota_H', 'iota_F']
    for par in pars:
        model.parameters.update({par: np.mean(samples_dict[par])})
    # set correct measures
    model.parameters.update({'economic_closures': economic_closures,
                             'telework': telework,
                             'social_restrictions': social_restrictions})
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
    # append to lists
    inhabitants.append(inhabitant)
    models.append(model)

##############
## simulate ##
##############

print(f"starting simulations")

# pre-made dataframe with desired formatting
lvl = [countries, length_measures_list] + [pd.date_range(start=start_simulation, end=end_simulation, freq='D'), ]
index = pd.MultiIndex.from_product(lvl, names=['country', 'length_measures', 'date'])     
outputs = pd.DataFrame(0, index=index, columns=states, dtype=float)     

# simulation loop
for i, length_measures in enumerate(length_measures_list):
    print(f"length of measures: {length_measures}")
    for country, model, inhabitant, trigger in zip(countries, models, inhabitants, triggers):
        # set correct length of measures and trigger
        model.parameters.update({'length_measures': length_measures,
                                 'trigger': trigger})
        # simulate
        simout = model.sim([start_simulation, end_simulation], rtol=1e-4)
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
            # normalise with initial amount if eco state 
            elif state in states_eco:
                out_copy /= out_copy.isel(date=0).values
                out_copy *= 100
            # add to output
            outputs.loc[(country,length_measures, slice(None)), states[j]] = out_copy.values
    # save output
    outputs.to_csv(f'simulations-prevention_paradox.csv')