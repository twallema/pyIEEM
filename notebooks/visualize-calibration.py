import random
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from matplotlib.ticker import MaxNLocator

from pyIEEM.data.data import get_economic_data, get_hospitalisation_incidence
from pyIEEM.models.utils import initialize_epinomic_model, aggregate_Brussels_Brabant_DataArray, dummy_aggregation

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

# simulation
N = 18
processes = 18
# visualisation (epi only + spatial)
n_draws_per_sample = 200
overdispersion_spatial = 0.047
overdispersion_national = 0.036
confint = 0.05
nrows = 3
ncols = 4
alpha_model_prediction = (0.05*18)/N
end_visualisation_epi = datetime(2021, 7, 1)
# visualiation (epi + eco national)
end_visualisation_eco = datetime(2021, 3, 1)
# compute simulation time
end_simulation = max(end_visualisation_epi, end_visualisation_eco)

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

# load data BE and SWE
data_BE_epi = get_hospitalisation_incidence('BE', aggregate_bxl_brabant=True)
data_SWE_epi = get_hospitalisation_incidence('SWE')

# load economic data BE and SWE
data_BE_eco_GDP = get_economic_data('GDP', 'BE', relative=True)
data_BE_eco_employment = get_economic_data('employment', 'BE', relative=True)
data_SWE_eco_GDP = get_economic_data('GDP', 'SWE', relative=True)
data_SWE_eco_employment = get_economic_data('employment', 'SWE', relative=True)

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
model_SWE = initialize_epinomic_model(
    'SWE', age_classes, True, start_calibration)

# load number of demographies
demographies=[]
for country in ['BE', 'SWE']:
    demographies.append(pd.read_csv(os.path.join(
                        abs_dir, f'../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).groupby(by='spatial_unit').sum().squeeze())

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
    param_dict['iota_H'] = samples_dict['iota_H'][i]
    param_dict['iota_F'] = samples_dict['iota_F'][i]
    return param_dict

#####################
## simulate models ##
#####################

outputs = []
for model in [model_BE, model_SWE]:
    outputs.append(model.sim([start_calibration, end_simulation], N=N,
                   processes=processes, draw_function=draw_function, samples=samples_dict))

################################################
## visualise calibration (epi + eco national) ##
################################################

titles = ['Belgium', 'Sweden']
countries = ['BE', 'SWE']

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8.3, 11.7*(2/3)), sharex=True)

for i, (out, data_epi, data_eco_GDP, data_eco_employment, country, demography) in enumerate(zip(outputs, [data_BE_epi, data_SWE_epi], [data_BE_eco_GDP, data_SWE_eco_GDP], [data_BE_eco_employment, data_SWE_eco_employment], ['BE', 'SWE'], demographies)):
    
    ## epidemiological
    data_calibration = data_epi.loc[slice(start_calibration, end_calibration_epi)].groupby(by='date').sum()/demography.sum()*100000
    data_post_calibration = data_epi.loc[slice(end_calibration_epi+timedelta(days=1), end_visualisation_eco)].groupby(by='date').sum()/demography.sum()*100000
    # data
    if country == 'BE':
        alpha = 0.6
        over = overdispersion_national
    else:
        alpha = 0.8
        over = 1e-3
    ax[0, i].scatter(data_calibration.index, data_calibration,
                    edgecolors='black', facecolors='black', marker='o', s=10, alpha=alpha)
    ax[0, i].scatter(data_post_calibration.index, data_post_calibration,
                    edgecolors='red', facecolors='red', marker='o', s=10, alpha=alpha)
    # model: add observational noise
    df_2plot = output_to_visuals(out, ['Hin',], n_draws_per_sample=n_draws_per_sample, alpha=over, LL = confint/2, UL = 1 - confint/2)
    # model: visualise
    for k in range(N):
        ax[0, i].plot(out.date, out.Hin.sum(dim=['age_class','spatial_unit']).isel(draws=k)/demography.sum()*100000, color='blue', linewidth=1.5, alpha=alpha_model_prediction)
    ax[0, i].fill_between(out.date, df_2plot['Hin', 'lower']/demography.sum()*100000, df_2plot['Hin', 'upper']/demography.sum()*100000, color='blue', alpha=0.2)
    # axes properties
    ax[0, i].set_xlim([start_calibration, end_visualisation_eco])
    ax[0, i].set_ylim([0, 8.2])
    ax[0, i].yaxis.set_major_locator(MaxNLocator(6))
    ax[0, i].set_title(titles[i])

    ### economic
    ## GDP
    data_calibration = data_eco_GDP.loc[slice(start_calibration, end_calibration_eco)]
    data_post_calibration = data_eco_GDP.loc[slice(end_calibration_eco+timedelta(days=1), end_visualisation_eco)]
    # data
    ax[1, i].scatter(data_calibration.index, 100*data_calibration,
                    edgecolors='black', facecolors='black', marker='o', s=10, alpha=0.8)
    ax[1, i].scatter(data_post_calibration.index, 100*data_post_calibration,
                    edgecolors='red', facecolors='red', marker='o', s=10, alpha=0.8)
    # model
    x_0 = out.x.sum(dim='NACE64').mean(dim='draws').isel(date=0).values
    for k in range(N):
        ax[1, i].plot(out.date, 100*out.x.sum(dim='NACE64').isel(draws=k)/x_0,  color='blue', linewidth=1.5, alpha=alpha_model_prediction)

    ax[1, i].fill_between(out.date, 100*out.x.sum(dim='NACE64').quantile(dim='draws', q=confint/2)/x_0,
                                    100*out.x.sum(dim='NACE64').quantile(dim='draws', q=1-confint/2)/x_0, color='blue', alpha=0.2)
    # axes properties
    ax[1, i].set_xlim([start_calibration, end_visualisation_eco])
    ax[1, i].set_ylim([60, 102])

    ## employment
    data_calibration = data_eco_employment.loc[slice(start_calibration, end_calibration_eco)]
    data_post_calibration = data_eco_employment.loc[slice(end_calibration_eco+timedelta(days=1), end_visualisation_eco)]
    # data
    ax[2, i].scatter(data_calibration.index, 100*data_calibration,
                    edgecolors='black', facecolors='black', marker='o', s=10, alpha=0.8)
    ax[2, i].scatter(data_post_calibration.index, 100*data_post_calibration,
                    edgecolors='red', facecolors='red', marker='o', s=10, alpha=0.8)
    # model
    l_0 = out.l.sum(dim='NACE64').mean(dim='draws').isel(date=0).values
    for k in range(N):
        ax[2, i].plot(out.date, 100*out.l.sum(dim='NACE64').isel(draws=k)/l_0,  color='blue', linewidth=1.5, alpha=alpha_model_prediction)
    ax[2, i].fill_between(out.date, 100*out.l.sum(dim='NACE64').quantile(dim='draws', q=confint/2)/l_0,
                                    100*out.l.sum(dim='NACE64').quantile(dim='draws', q=1-confint/2)/l_0, color='blue', alpha=0.2)
    # axes properties
    ax[2, i].set_xlim([start_calibration, end_visualisation_eco])
    ax[2, i].set_ylim([60, 102])
    # rotate labels
    for tick in ax[2,i].get_xticklabels():
        tick.set_rotation(30)
    # ylabels left hand side only
    if i == 0:
        ax[0, i].set_ylabel('Hospital incidence\nper 100K inhab. (-)')
        ax[1, i].set_ylabel('Gross aggregated\noutput (%)')
        ax[2, i].set_ylabel('Labor compensation (%)')

plt.savefig(
    f'calibration_epinomic_national.png', dpi=600)
#plt.show()
plt.close()

###########################################
## visualise calibration (epi + spatial) ##
###########################################

ylimits = [[0,13],[0,8]]
aggregation_functions = [aggregate_Brussels_Brabant_DataArray, dummy_aggregation]

# visualisation
for out, data, country, aggfunc, demography, ylimit in zip(outputs, [data_BE_epi, data_SWE_epi], countries, aggregation_functions, demographies, ylimits):

    # aggregate model
    out = aggfunc(out.Hin)

    # get dates calibration
    dates_calibration = data.loc[slice(start_calibration, end_calibration_epi), slice(
        None)].index.get_level_values('date').unique()
    dates_post_calibration = data.loc[slice(end_calibration_epi+timedelta(
        days=1), end_visualisation_epi), slice(None)].index.get_level_values('date').unique()
    spatial_units = data.index.get_level_values('spatial_unit').unique()
    
    # visualise
    n_figs = 0
    counter = 0
    while counter <= len(spatial_units):
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(11.7, 8.3), sharex=True, sharey=True)
        axes = axes.flatten()
        for j, ax in enumerate(axes):
            if j+counter <= len(spatial_units):
                if j + counter < len(spatial_units):
                    # extract demographics
                    if spatial_units[j+counter] == 'Brussels and Brabant':
                            demo = demography.loc['Brussels'] + demography.loc['Vlaams-Brabant'] + demography.loc['Brabant Wallon']
                    else:
                        demo = demography.loc[spatial_units[j+counter]]
                    # plot data
                    ax.scatter(dates_calibration, data.loc[slice(start_calibration, end_calibration_epi), spatial_units[j+counter]]/demo*100000,
                               edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.6)
                    ax.scatter(dates_post_calibration, data.loc[slice(end_calibration_epi+timedelta(days=1), end_visualisation_epi), spatial_units[j+counter]]/demo*100000,
                               edgecolors='red', facecolors='white', marker='o', s=10, alpha=0.6)
                    # plot model prediction
                    for k in range(N):
                        ax.plot(out.date, out.sel(spatial_unit=spatial_units[j+counter]).sum(
                        dim='age_class').isel(draws=k)/demo*100000, color='blue', linewidth=1.5, alpha=alpha_model_prediction)
                    if country == 'BE':
                        ov = overdispersion_spatial
                    else:
                        ov = 1e-3
                    df_2plot = output_to_visuals(out.sel(spatial_unit=spatial_units[j+counter]).to_dataset(name='Hin'), ['Hin',], n_draws_per_sample=n_draws_per_sample, alpha=ov, LL = confint/2, UL = 1 - confint/2)
                    ax.fill_between(out.date, df_2plot['Hin', 'lower']/demo*100000, df_2plot['Hin', 'upper']/demo*100000, color='blue', alpha=0.2)
                    # shade VOCs and vaccines
                    ax.axvspan('2021-02-01', end_visualisation_epi,
                               color='black', alpha=0.1)
                    # set title
                    ax.set_title(spatial_units[j+counter])
                else:
                    # plot data
                    ax.scatter(dates_calibration, data.groupby(by='date').sum().loc[slice(start_calibration, end_calibration_epi)]/demography.sum()*100000,
                               edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.6)
                    ax.scatter(dates_post_calibration, data.groupby(by='date').sum().loc[slice(end_calibration_epi+timedelta(days=1), end_visualisation_epi)]/demography.sum()*100000,
                               edgecolors='red', facecolors='white', marker='o', s=10, alpha=0.6)
                    # plot model prediction
                    for k in range(N):
                        ax.plot(out.date, out.sum(dim=['age_class', 'spatial_unit']).sel(
                        draws=k)/demography.sum()*100000, color='blue', linewidth=1.5, alpha=alpha_model_prediction)
                    if country == 'BE':
                        ov = overdispersion_national
                    else:
                        ov = 1e-3
                    df_2plot = output_to_visuals(out.to_dataset(name='Hin'), ['Hin',], n_draws_per_sample=n_draws_per_sample, alpha=ov, LL = confint/2, UL = 1 - confint/2)
                    ax.fill_between(out.date, df_2plot['Hin', 'lower']/demography.sum()*100000, df_2plot['Hin', 'upper']/demography.sum()*100000, color='blue', alpha=0.2)
                    # shade VOCs and vaccines + text
                    ax.axvspan('2021-02-01', end_visualisation_epi,
                               color='black', alpha=0.1)
                    ax.text(0.72, 0.85, 'Vaccination\n   Variants',
                            transform=ax.transAxes, fontsize=6)
                    # set title
                    ax.set_title(country)
                # set maximum number of xlabels
                ax.xaxis.set_major_locator(MaxNLocator(7))
                # rotate labels
                for tick in ax.get_xticklabels():
                    tick.set_rotation(90)
                # limit xrange
                ax.set_xlim([start_calibration, end_visualisation_epi])
                # set ylabel
                if j % ncols == 0:
                    ax.set_ylabel('Hospital incidence\nper 100K inhab. (-)')
                # set ylim
                ax.set_ylim(ylimit)
            else:
                fig.delaxes(ax)
        n_figs += 1
        counter += nrows*ncols
        plt.savefig(
            f'calibration_epinomic_{country}_part_{n_figs}.png', dpi=600)
        # plt.show()
        plt.close()
