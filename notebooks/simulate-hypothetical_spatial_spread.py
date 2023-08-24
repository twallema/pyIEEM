import os
import json
import random
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

weigh_demographic = True
countries = ['SWE', 'BE']
# where to drop initial condition
spatial_units_always = [['Stockholm'], ['Brussels',],]
# simulation
start_simulation = datetime(2020, 2, 1)
end_simulation = datetime(2020, 7, 1)
states_epi = ['Hin', 'Ih']
states_eco = ['x', 'l']
# visualisation (epi only + spatial)
confint = 0.05

#################################################
## Helper function to change initial condition ##
#################################################

# helper function to adjust initial condition
def update_initial_condition(spatial_units_additional, spatial_units_always, demography, spatial_units, initial_condition, N, weigh_demographic=True):
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
    # use demography to distribute contacts or not
    if weigh_demographic==False:
        # sum of all infected is always one, only spatial distribution is altered
        infected = 1/(len(spatial_units_additional) + len(spatial_units_always))
        # loop over spatial patches you always want to put an infected in
        for j, patchname in enumerate(spatial_units):
            # check if always present
            if patchname in spatial_units_always:
                inf = infected * (N[:, j]/sum(N[:, j]))
                output[:, j] += inf
            # check if additional spatial patch
            if patchname in spatial_units_additional:
                inf = infected * (N[:, j]/sum(N[:, j]))
                output[:, j] += inf    
    else:
        # compute total number of inhabitants
        total_inhab=0
        for su in spatial_units_always+spatial_units_additional:
            total_inhab += demography.loc[su]
        # divide one infected over the spatial units
        infected = demography.loc[spatial_units_always+spatial_units_additional]/total_inhab
        for j, patchname in enumerate(spatial_units):
            if patchname in spatial_units_always:
                if isinstance(infected.loc[patchname], np.float64):
                    inf = infected.loc[patchname] * (N[:, j]/sum(N[:, j]))
                else:
                    inf = infected.loc[patchname].unique() * (N[:, j]/sum(N[:, j]))
                output[:, j] += inf 
            if patchname in spatial_units_additional:
                if isinstance(infected.loc[patchname], np.float64):
                    inf = infected.loc[patchname] * (N[:, j]/sum(N[:, j]))
                else:
                    inf = infected.loc[patchname].unique() * (N[:, j]/sum(N[:, j]))
                output[:, j] += inf    
    return output

#########################
## load model and data ##
#########################

## start loop here
for i,country in enumerate(countries):
    print(f"\nworking on country: {country}")
    # load samples dictionary
    samples_dict = json.load(open(args.identifier+'_SAMPLES_'+args.date+'.json'))
    # load model SWE with default initial condition
    age_classes = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (
        35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 120)], closed='left')
    model = initialize_epinomic_model(country, age_classes, True, simulation_start=start_simulation, scenarios=f'hypothetical_spatial_spread')
    # load number of inhabitants for normalisation of data
    inhabitants = pd.read_csv(os.path.join(abs_dir, f'../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).sum().values[0]
    # load full demography to distribute initial infected
    demography = pd.read_csv(os.path.join(
        abs_dir, f'../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).groupby(by='spatial_unit').sum().squeeze()
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
    # pre-allocate dataframe with desired formatting
    index_names = ['spatial_unit', 'date']       
    dates = pd.date_range(start=start_simulation, end=end_simulation, freq='D')
    spatial_units = model.coordinates['spatial_unit']
    index = pd.MultiIndex.from_product([spatial_units, dates], names=index_names)
    outputs = pd.DataFrame(0, index=index, columns=states_epi+states_eco, dtype=float)                           

    ################
    ## simulation ##
    ################

    # simulation loop
    for spatial_unit in spatial_units:
        print(f"\tworking on spatial unit: '{spatial_unit}'")
        # adjust initial condition
        init_E = update_initial_condition([spatial_unit,], spatial_units_always[i], demography, model.coordinates['spatial_unit'],
                                            model.initial_states['E'], model.parameters['N'], weigh_demographic)
        model.initial_states.update({'E': init_E})
        # simulate model
        simout = model.sim([start_simulation, end_simulation])
        # interpolate to a daily timestep
        simout = simout.interp({'date': pd.date_range(start=start_simulation, end=end_simulation, freq='D')}, method="linear")
        for state in states_epi+states_eco:
            # aggregate and append epi states
            simout_copy = simout[state].copy()
            for dim in simout[state].dims:
                if dim != 'date':
                    simout_copy = simout_copy.sum(dim=dim)
            # normalise
            if state in states_epi:        
                outputs.loc[(spatial_unit, slice(None)), state] = simout_copy/inhabitants*100000
            if state in states_eco:
                outputs.loc[(spatial_unit, slice(None)), state] = simout_copy/simout_copy.isel(date=0)*100
        # save output
        outputs.to_csv(f'simulations-hypothetical_spatial_spread_{country}-demographic_weighing_{weigh_demographic}.csv')