import random
import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from matplotlib.ticker import MaxNLocator

from pySODM.optimization.utils import add_negative_binomial_noise
from pyIEEM.data.data import get_economic_data, get_hospitalisation_incidence
from pyIEEM.models.utils import initialize_epinomic_model, aggregate_Brussels_Brabant_Dataset, dummy_aggregation

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
N = 2
processes = 2
# visualisation (epi only + spatial)
confint = 0.05
nrows = 3
ncols = 4
end_visualisation_epi = datetime(2021, 7, 1)
# visualiation (epi + eco national)
end_visualisation_eco = datetime(2021, 3, 1)
# compute simulation time
end_simulation = max(end_visualisation_epi, end_visualisation_eco)

#########################
## load model and data ##
#########################

# load data BE and SWE
data_BE_epi = get_hospitalisation_incidence('BE', aggregate_bxl_brabant=True)
data_SWE_epi = get_hospitalisation_incidence('SWE')

# load economic data BE and SWE
data_BE_eco = get_economic_data('GDP', 'BE')
data_SWE_eco = get_economic_data('GDP', 'SWE')

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

end_calibration_eco = end_calibration_epi = end_simulation = datetime(2020, 6, 1)

##########################
## define draw function ##
##########################

def draw_function(param_dict, samples_dict):
    i, param_dict['nu'] = random.choice(list(enumerate(samples_dict['nu'])))
    param_dict['xi_eff'] = samples_dict['xi_eff'][i]
    param_dict['pi_eff'] = samples_dict['pi_eff'][i]
    param_dict['pi_work'] = samples_dict['pi_work'][i]
    param_dict['pi_leisure'] = samples_dict['pi_leisure'][i]
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

for out, data_epi, data_eco, country in zip(outputs, [data_BE_epi, data_SWE_epi], [data_BE_eco, data_SWE_eco], ['BE', 'SWE']):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11.7, 8.3), sharex=True)

    ## epidemiological
    data_calibration = data_epi.loc[slice(start_calibration, end_calibration_epi)].groupby(by='date').sum()
    data_post_calibration = data_epi.loc[slice(end_calibration_epi+timedelta(days=1), end_visualisation_eco)].groupby(by='date').sum()
    # data
    ax[0].scatter(data_calibration.index, data_calibration,
                    edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.6)
    ax[0].scatter(data_post_calibration.index, data_post_calibration,
                    edgecolors='red', facecolors='white', marker='o', s=10, alpha=0.6)
    # model: add observational noise
    out_obs = add_negative_binomial_noise(out.Hin.to_dataset(), alpha=0.027)
    # model: visualise
    ax[0].plot(out.date, out_obs.Hin.sum(dim=['age_class','spatial_unit']).mean(dim='draws'), color='blue', linewidth=1)
    ax[0].fill_between(out_obs.date, out_obs.Hin.sum(dim=['age_class','spatial_unit']).quantile(dim='draws', q=confint/2),
                                    out_obs.Hin.sum(dim=['age_class','spatial_unit']).quantile(dim='draws', q=1-confint/2), color='blue', alpha=0.2)
    # axes properties
    ax[0].set_xlim([start_calibration, end_visualisation_eco])
    ax[0].set_ylim([0, 850])
    ax[0].set_ylabel('Hospital incidence (-)')

    ## economic
    data_calibration = data_eco.loc[slice(start_calibration, end_calibration_eco)].groupby(by='date').sum()
    data_post_calibration = data_eco.loc[slice(end_calibration_eco+timedelta(days=1), end_visualisation_eco)].groupby(by='date').sum()
    # data countries
    x_0 = out.x.sum(dim='NACE64').mean(dim='draws').isel(date=0).values
    ax[1].scatter(data_calibration.index, 100*data_calibration/x_0,
                    edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.6)
    ax[1].scatter(data_post_calibration.index, 100*data_post_calibration/x_0,
                    edgecolors='red', facecolors='white', marker='o', s=10, alpha=0.6)
    # model
    ax[1].plot(out.date, 100*out.x.sum(dim='NACE64').mean(dim='draws')/x_0,  color='blue', linewidth=1)
    ax[1].fill_between(out.date, 100*out.x.sum(dim='NACE64').quantile(dim='draws', q=confint/2)/x_0,
                                    100*out.x.sum(dim='NACE64').quantile(dim='draws', q=1-confint/2)/x_0, color='blue', alpha=0.2)
    # axes properties
    ax[1].set_xlim([start_calibration, end_visualisation_eco])
    #ax[1].set_ylim([60, 105])
    ax[1].set_ylabel('Productivity loss (%)')

    plt.savefig(
        f'calibration_epinomic_national_{country}.png', dpi=400)
    # plt.show()
    plt.close()

###########################################
## visualise calibration (epi + spatial) ##
###########################################

aggregation_functions = [aggregate_Brussels_Brabant_Dataset, dummy_aggregation]
countries = ['BE', 'SWE']

# visualisation
for output, data, country, aggfunc in zip(outputs, [data_BE_epi, data_SWE_epi], countries, aggregation_functions):

    # add observational noise
    out_obs = add_negative_binomial_noise(output.Hin.to_dataset(), alpha=0.027)
    out_nat_obs = add_negative_binomial_noise(
        output.sum(dim='spatial_unit').Hin.to_dataset(), alpha=0.027)

    # aggregate model
    out_obs = aggfunc(out_obs)

    # slice hospitalisation states
    out_obs = out_obs.Hin
    out_nat_obs = out_nat_obs.Hin

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
            nrows=nrows, ncols=ncols, figsize=(11.7, 8.3), sharex=True)
        axes = axes.flatten()
        for j, ax in enumerate(axes):
            if j+counter <= len(spatial_units):
                if j + counter < len(spatial_units):
                    # plot data
                    ax.scatter(dates_calibration, data.loc[slice(start_calibration, end_calibration_epi), spatial_units[j+counter]],
                               edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.6)
                    ax.scatter(dates_post_calibration, data.loc[slice(end_calibration_epi+timedelta(days=1), end_visualisation_epi), spatial_units[j+counter]],
                               edgecolors='red', facecolors='white', marker='o', s=10, alpha=0.6)
                    # plot model prediction
                    ax.plot(out_obs.date, out_obs.sel(spatial_unit=spatial_units[j+counter]).sum(
                        dim='age_class').mean(dim='draws'), color='blue', linewidth=1)
                    ax.fill_between(out_obs.date, out_obs.sel(spatial_unit=spatial_units[j+counter]).sum(dim='age_class').quantile(dim='draws', q=confint/2),
                                    out_obs.sel(spatial_unit=spatial_units[j+counter]).sum(dim='age_class').quantile(dim='draws', q=1-confint/2), color='blue', alpha=0.2)
                    # shade VOCs and vaccines
                    ax.axvspan('2021-02-01', end_visualisation_epi,
                               color='black', alpha=0.1)
                    # set title
                    ax.set_title(spatial_units[j+counter])
                else:
                    # plot data
                    ax.scatter(dates_calibration, data.groupby(by='date').sum().loc[slice(start_calibration, end_calibration_epi)],
                               edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.6)
                    ax.scatter(dates_post_calibration, data.groupby(by='date').sum().loc[slice(end_calibration_epi+timedelta(days=1), end_visualisation_epi)],
                               edgecolors='red', facecolors='white', marker='o', s=10, alpha=0.6)
                    # plot model prediction
                    ax.plot(out_nat_obs.date, out_nat_obs.sum(dim=['age_class']).mean(
                        dim='draws'), color='blue', linewidth=1)
                    ax.fill_between(out_nat_obs.date, out_nat_obs.sum(dim=['age_class']).quantile(dim='draws', q=confint/2),
                                    out_nat_obs.sum(dim=['age_class']).quantile(dim='draws', q=1-confint/2), color='blue', alpha=0.2)
                    # shade VOCs and vaccines + text
                    ax.axvspan('2021-02-01', end_visualisation_epi,
                               color='black', alpha=0.1)
                    ax.text(0.72, 0.85, 'Vaccination\n   Variants',
                            transform=ax.transAxes, fontsize=6)
                    # set title
                    ax.set_title(country)
                # set maximum number of labels
                ax.xaxis.set_major_locator(MaxNLocator(5))
                # rotate labels
                for tick in ax.get_xticklabels():
                    tick.set_rotation(60)
                # limit xrange
                ax.set_xlim([start_calibration, end_visualisation_epi])
            else:
                fig.delaxes(ax)
        n_figs += 1
        counter += nrows*ncols
        plt.savefig(
            f'calibration_epinomic_{country}_part_{n_figs}.png', dpi=400)
        # plt.show()
        plt.close()
