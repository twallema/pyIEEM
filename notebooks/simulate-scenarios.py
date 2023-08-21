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

scenarios = ['L1', 'L4']
t_start_lockdowns = [datetime(2020, 3, 15), datetime(2020, 3, 12)]
l_economics = [7,]
# simulation
N = 2
processes = 2
start_simulation = datetime(2020, 3, 1)
end_simulation = datetime(2020, 5, 1)
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
start_calibration = datetime.strptime(
    samples_dict['start_calibration'], '%Y-%m-%d')
end_calibration_epi = datetime.strptime(
    samples_dict['end_calibration_epi'], '%Y-%m-%d')
end_calibration_eco = datetime.strptime(
    samples_dict['end_calibration_eco'], '%Y-%m-%d')    

# load model BE and SWE
age_classes = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (
    35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 120)], closed='left')
model_BE = initialize_epinomic_model(
    'BE', age_classes, True, start_calibration)

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

# pre-allocate a pd.Dataframe for the simulation output

scenarios = ['L1', 'L4']
t_start_lockdowns = [datetime(2020, 3, 15), datetime(2020, 3, 12)]
l_economics = [7,]

# pre-made dataframe with desired formatting
names = ['sector', 'type_day', 'vacation']
sectors = d.reset_index()['sector'].unique()
type_days = d.reset_index()['type_day'].unique()
vacations = d.reset_index()['vacation'].unique()                  
iterables = [sectors, type_days, vacations]
ratios = pd.Series(index=pd.MultiIndex.from_product(iterables, names=names), name='ratio_contacts', dtype=float)                           


# simulate
out = model_BE.sim([start_simulation, end_simulation], N=N,
                processes=processes, draw_function=draw_function, samples=samples_dict)

# save output