import random
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from matplotlib.ticker import MaxNLocator
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

scenarios = ['L1', 'L2b', 'L3b', 'L4a', 'L4b']
t_start_lockdowns = [datetime(2020, 3, 18), datetime(2020, 3, 15), datetime(2020, 3, 12), datetime(2020, 3, 9), datetime(2020, 3, 6), datetime(2020, 3, 3)]
for i,val in enumerate(t_start_lockdowns):
    t_start_lockdowns[i] = val + timedelta(days=1)
# simulation
N = 3
processes = 3
start_simulation = datetime(2020, 3, 1)
end_simulation = datetime(2021, 3, 1)
states_epi = ['Hin', 'Ih']
states_eco = ['x', 'l']
# visualisation (epi only + spatial)
confint = 0.05

#########################
## load model and data ##
#########################

# load samples dictionary
samples_dict = json.load(open(args.identifier+'_SAMPLES_'+args.date+'.json'))

# load model BE
age_classes = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (
    35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 120)], closed='left')
model = initialize_epinomic_model('BE', age_classes, True, start_simulation, scenarios='hypothetical_policy')
# load number of inhabitants
inhabitants = pd.read_csv(os.path.join(abs_dir, f'../data/interim/epi/demographic/age_structure_BE_2019.csv'), index_col=[0, 1]).sum().values[0]

##########################
## define draw function ##
##########################

def draw_function(param_dict, samples_dict):
    i, param_dict['nu'] = random.choice(list(enumerate(samples_dict['nu'])))
    param_dict['xi_eff'] = samples_dict['xi_eff'][i]
    param_dict['pi_eff'] = samples_dict['pi_eff'][i]
    param_dict['pi_work'] = samples_dict['pi_work'][i]
    param_dict['pi_leisure'] = samples_dict['pi_leisure'][i]
    param_dict['mu'] = samples_dict['mu'][i]
    param_dict['amplitude_BE'] = samples_dict['amplitude_BE'][i]
    param_dict['amplitude_SWE'] = samples_dict['amplitude_SWE'][i]
    param_dict['peak_shift_BE'] = samples_dict['peak_shift_BE'][i]
    param_dict['peak_shift_SWE'] = samples_dict['peak_shift_SWE'][i]
    param_dict['iota_F'] = samples_dict['iota_F'][i]
    param_dict['iota_H'] = samples_dict['iota_H'][i]
    return param_dict

########################
## simulate scenarios ##
########################

# pre-made dataframe with desired formatting
index_names = ['scenario', 't_start_lockdown', 'date']    
column_names = ['state', 'statistic']        
index=pd.MultiIndex.from_product([scenarios, t_start_lockdowns, pd.date_range(start=start_simulation, end=end_simulation, freq='D')], names=index_names)
columns=pd.MultiIndex.from_product([states_epi+states_eco, ['mean', 'median', 'lower', 'upper']], names=column_names)
outputs = pd.DataFrame(0, index=index, columns=columns, dtype=float)                           

# simulate model
for scenario in scenarios:
    model.parameters.update({'scenario': scenario})
    print(f'working on scenario {scenario}')
    for t_start_lockdown in t_start_lockdowns:
        # set right scenario parameters
        model.parameters.update({'t_start_lockdown': t_start_lockdown})
        print(f'\tlockdown starts: {t_start_lockdown}')
        # simulate model
        simout = model.sim([start_simulation, end_simulation], N=N, processes=processes, draw_function=draw_function, samples=samples_dict)
        # interpolate to required daterange
        simout = simout.interp({'date': pd.date_range(start=start_simulation, end=end_simulation, freq='D')}, method="linear")
        ## epidemiological states
        # compute mean, median, lower and upper
        for state_epi in states_epi:
            outputs.loc[(scenario, t_start_lockdown, slice(None)), (state_epi, 'mean')] = simout[state_epi].sum(dim=['age_class', 'spatial_unit']).mean(dim='draws').values/inhabitants*100000
            outputs.loc[(scenario, t_start_lockdown, slice(None)), (state_epi, 'median')] = simout[state_epi].sum(dim=['age_class', 'spatial_unit']).median(dim='draws').values/inhabitants*100000
            outputs.loc[(scenario, t_start_lockdown, slice(None)), (state_epi, 'lower')] = simout[state_epi].sum(dim=['age_class', 'spatial_unit']).quantile(dim='draws', q=confint/2).values/inhabitants*100000
            outputs.loc[(scenario, t_start_lockdown, slice(None)), (state_epi, 'upper')] = simout[state_epi].sum(dim=['age_class', 'spatial_unit']).quantile(dim='draws', q=1-confint/2).values/inhabitants*100000
        ## economic states                         
        # compute mean, median, lower and upper
        for state_eco in states_eco:
            outputs.loc[(scenario, t_start_lockdown, slice(None)), (state_eco, 'mean')] = 100*(simout[state_eco].sum(dim=['NACE64']).mean(dim='draws')/simout[state_eco].sum(dim=['NACE64']).mean(dim='draws').isel(date=0)).values
            outputs.loc[(scenario, t_start_lockdown, slice(None)), (state_eco, 'median')] = 100*(simout[state_eco].sum(dim=['NACE64']).median(dim='draws')/simout[state_eco].sum(dim=['NACE64']).mean(dim='draws').isel(date=0)).values
            outputs.loc[(scenario, t_start_lockdown, slice(None)), (state_eco, 'lower')] = 100*(simout[state_eco].sum(dim=['NACE64']).quantile(dim='draws', q=confint/2)/simout[state_eco].sum(dim=['NACE64']).mean(dim='draws').isel(date=0)).values
            outputs.loc[(scenario, t_start_lockdown, slice(None)), (state_eco, 'upper')] = 100*(simout[state_eco].sum(dim=['NACE64']).quantile(dim='draws', q=1-confint/2)/simout[state_eco].sum(dim=['NACE64']).mean(dim='draws').isel(date=0)).values

# save output
outputs.to_csv('simulations_hypothetical_scenarios_BE.csv')