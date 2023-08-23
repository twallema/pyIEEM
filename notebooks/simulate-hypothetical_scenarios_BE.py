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

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

abs_dir = os.path.dirname(__file__)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-ID", "--identifier", help="Calibration identifier")
parser.add_argument("-d", "--date", help="Calibration date")
args = parser.parse_args()

##########################
## change settings here ##
##########################

scenarios = ['L1', 'L2a', 'L2b', 'L3a', 'L3b', 'L4a', 'L4b']
t_start_lockdowns = [datetime(2020, 3, 15), datetime(2020, 3, 13)]
l_economics = [7,]
# simulation
N = 6
processes = 6
start_simulation = datetime(2020, 3, 1)
end_simulation = datetime(2020, 9, 1)
states_epi = ['Hin', 'Ih']
states_eco = ['x', 'l']
# visualisation (epi only + spatial)
n_draws_per_sample = 200
overdispersion = 0.036
confint = 0.05

#############################################
## Helper function for observational noise ##
#############################################

def output_to_visuals(output, states, alpha=1e-6, n_draws_per_sample=1, UL=1-0.05*0.5, LL=0.05*0.5):
    """
    A function to add the a-posteriori poisson uncertainty on the relationship between the model output and data
    and format the model output in a pandas dataframe for easy acces


    Parameters
    ----------

    output : xarray
        Simulation output xarray

    states : xarray
        Model states on which to add the a-posteriori poisson uncertainty

    alpha: float
        Overdispersion factor of the negative binomial distribution. For alpha --> 0, the negative binomial converges to the poisson distribution.

    n_draws_per_sample : int
        Number of poisson experiments to be added to each simulated trajectory (default: 1)

    UL : float
        Upper quantile of simulation result (default: 97.5%)

    LL : float
        Lower quantile of simulation result (default: 2.5%)

    Returns
    -------

    df : pd.DataFrame
        contains for every model state the mean, median, lower- and upper quantiles
        index is equal to simtime

    Example use
    -----------

    simtime, df_2plot = output_to_visuals(output, 100, 1, LL = 0.05/2, UL = 1 - 0.05/2)
    # x-values do not need to be supplied when using `plt.plot`
    plt.plot(df_2plot['H_in', 'mean'])
    # x-values must be supplied when using `plt.fill_between`
    plt.fill_between(simtime, df_2plot['H_in', 'LL'], df_2plot['H_in', 'UL'])

    """
    # Check if dimension draws is present
    if not 'draws' in list(output.dims):
        raise ValueError(
            "dimension 'draws' is not present in model output xarray"
        )
    # Check if the states are present
    for state_name in states:
        if not state_name in list(output.data_vars):
            raise ValueError(
                "variable state_name '{0}' is not a model state".format(state_name)
            )
    # Initialize a pandas dataframe for results
    columns = [[],[]]
    tuples = list(zip(*columns))
    columns = pd.MultiIndex.from_tuples(tuples, names=["model state", "quantity"])
    df = pd.DataFrame(index=pd.to_datetime(output['date'].values), columns=columns)
    df.index.name = 'simtime'
    # Deepcopy xarray output (it is mutable like a dictionary!)
    copy = output.copy(deep=True)
    # Loop over output states
    for state_name in states:
        # Automatically sum all dimensions except time and draws
        for dimension in output[state_name].dims:
            if ((dimension != 'date') & (dimension != 'draws')):
                copy[state_name] = copy[state_name].sum(dim=dimension)
        mean, median, lower, upper = add_negative_binomial(copy[state_name].values, alpha, n_draws_per_sample, UL, LL, add_to_mean=False)
        # Add to dataframe
        df[state_name,'mean'] = mean
        df[state_name,'median'] = median
        df[state_name,'lower'] = lower
        df[state_name,'upper'] = upper
    return df

def add_negative_binomial(output_array, alpha, n_draws_per_sample=100, UL=0.05*0.5, LL=1-0.05*0.5, add_to_mean=True):
    """ A function to add a-posteriori negative binomial uncertainty on the relationship between the model output and data
    
    Parameters
    ----------

    output_array: np.array
        2D numpy array containing the simulation result. First axis: draws, second axis: time.

    alpha: float
        Negative binomial overdispersion coefficient

    n_draws_per_sample: int
        Number of draws to take from the negative binomial distribution at each timestep and then average out.
    
    LL: float
        Lower quantile limit.

    UL: float
        Upper quantile limit.

    add_to_mean: boolean
        If True, `n_draws_per_sample` negative binomial draws are added to the mean model prediction. If False, `n_draws_per_sample` negative binomial draws are added to each of the `n_samples` model predictions.
        Both options converge for large `n_draws_per_sample`.

    Returns
    -------

    mean: np.array
        1D numpy array containing the mean model prediction at every timestep
    
    median: np.array
        1D numpy array containing the mean model prediction at every timestep
    
    lower: np.array
        1D numpy array containing the lower quantile of the model prediction at every timestep

    upper: np.array
        1D numpy array containing the upper quantile of the model prediction at every timestep
    """

    # Determine number of samples and number of timesteps
    simtime = output_array.shape[1]
    if add_to_mean:
        output_array= np.mean(output_array, axis=0)
        output_array=output_array[np.newaxis, :]
        n_samples=1
    else:
        n_samples = output_array.shape[0]
    # Initialize a column vector to append to
    vector = np.zeros((simtime,1))
    # Loop over dimension draws
    for n in range(n_samples):
        try:
            for draw in range(n_draws_per_sample):
                vector = np.append(vector, np.expand_dims(np.random.negative_binomial(1/alpha, (1/alpha)/(output_array[n,:] + (1/alpha)), size = output_array.shape[1]), axis=1), axis=1)
        except:
            warnings.warn("I had to remove a simulation result from the output because there was a negative value in it..")

    # Remove first column
    vector = np.delete(vector, 0, axis=1)
    #  Compute mean and median
    mean = np.mean(vector,axis=1)
    median = np.median(vector,axis=1)    
    # Compute quantiles
    lower = np.quantile(vector, q = LL, axis = 1)
    upper = np.quantile(vector, q = UL, axis = 1)

    return mean, median, lower, upper

#########################
## load model and data ##
#########################

# load samples dictionary
samples_dict = json.load(open(args.identifier+'_SAMPLES_'+args.date+'.json'))

# load model BE and SWE
age_classes = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (
    35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 120)], closed='left')
model = initialize_epinomic_model('BE', age_classes, True, start_simulation, scenarios=True)

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
    #param_dict['iota_F'] = samples_dict['iota_F'][i]
    #param_dict['iota_H'] = samples_dict['iota_H'][i]
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
        # aggregate to national nevel
        simout_epi = simout[states_epi].sum(dim=['age_class', 'spatial_unit'])
        # add observational noise and compute statistics
        simout_epi  = output_to_visuals(simout_epi, states_epi, n_draws_per_sample=n_draws_per_sample, alpha=overdispersion, LL = confint/2, UL = 1 - confint/2)
        # add to dataset
        outputs.loc[(scenario, t_start_lockdown, slice(None)), (states_epi, slice(None))] = simout_epi.values
        ## economic states
        # pre-allocate dataframe
        columns=pd.MultiIndex.from_product([states_eco, ['mean', 'median', 'lower', 'upper']], names=column_names)
        simout_eco = pd.DataFrame(0, index=pd.date_range(start=start_simulation, end=end_simulation, freq='D'), columns=columns, dtype=float)                           
        # compute mean, median, lower and upper
        for state_eco in states_eco:
            simout_eco.loc[slice(None), (state_eco, 'mean')] = (simout[state_eco].sum(dim=['NACE64']).mean(dim='draws')/simout[state_eco].sum(dim=['NACE64']).mean(dim='draws').isel(date=0)).values
            simout_eco.loc[slice(None), (state_eco, 'median')] = (simout[state_eco].sum(dim=['NACE64']).median(dim='draws')/simout[state_eco].sum(dim=['NACE64']).mean(dim='draws').isel(date=0)).values
            simout_eco.loc[slice(None), (state_eco, 'lower')] = (simout[state_eco].sum(dim=['NACE64']).quantile(dim='draws', q=confint/2)/simout[state_eco].sum(dim=['NACE64']).mean(dim='draws').isel(date=0)).values
            simout_eco.loc[slice(None), (state_eco, 'upper')] = (simout[state_eco].sum(dim=['NACE64']).quantile(dim='draws', q=1-confint/2)/simout[state_eco].sum(dim=['NACE64']).mean(dim='draws').isel(date=0)).values
        # concatenate dataframes
        outputs.loc[(scenario, t_start_lockdown, slice(None)), (states_eco, slice(None))] = simout_eco.values

# save output
outputs.to_csv('simulations_scenarios.csv')