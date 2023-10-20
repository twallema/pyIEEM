from pySODM.optimization.utils import add_negative_binomial_noise
from pyIEEM.data.data import get_hospitalisation_incidence
from pyIEEM.models.utils import initialize_epidemic_model, aggregate_Brussels_Brabant_Dataset, dummy_aggregation
from datetime import timedelta, datetime
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
abs_dir = os.path.dirname(__file__)

# parse arguments
import argparse
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
end_simulation = datetime.strptime('2021-07-01', '%Y-%m-%d')
# visualisation
alpha = 0.05
nrows = 3
ncols = 4

#########################
## load model and data ##
#########################

# load data BE and SWE
data_BE = get_hospitalisation_incidence('BE', aggregate_bxl_brabant=True)
data_SWE = get_hospitalisation_incidence('SWE')

# load samples dictionary
import json
samples_dict = json.load(open(args.identifier+'_SAMPLES_'+args.date+'.json'))
start_calibration = datetime.strptime(samples_dict['start_calibration'], '%Y-%m-%d')
end_calibration = datetime.strptime(samples_dict['end_calibration'], '%Y-%m-%d')

# load model BE and SWE
age_classes = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (
    35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 120)], closed='left')
model_BE = initialize_epidemic_model('BE', age_classes, True, start_calibration)
model_SWE = initialize_epidemic_model('SWE', age_classes, True, start_calibration)

##########################
## define draw function ##
##########################

import random
def draw_function(param_dict, samples_dict):
    i, param_dict['nu'] = random.choice(list(enumerate(samples_dict['tau'])))  
    param_dict['xi_eff'] = samples_dict['ypsilon_eff'][i]
    param_dict['pi_eff'] = samples_dict['phi_eff'][i]
    param_dict['pi_work'] = samples_dict['phi_work'][i]
    param_dict['pi_leisure'] = samples_dict['phi_leisure'][i]
    return param_dict

###########################
## visualise calibration ##
###########################

aggregation_functions = [aggregate_Brussels_Brabant_Dataset, dummy_aggregation]
models = [model_BE, model_SWE]
datasets = [data_BE, data_SWE]
countries = ['BE', 'SWE']

# visualisation
for model, data, country, aggfunc in zip(models, datasets, countries, aggregation_functions):

    # simulate model
    out = model.sim([start_calibration, end_simulation], N=N, processes=processes, draw_function=draw_function, samples=samples_dict)
    
    # aggregate model
    out = aggfunc(out)

    # add observational noise
    out_obs = add_negative_binomial_noise(out, alpha = 0.027)
    out_nat_obs = add_negative_binomial_noise(out.sum(dim='spatial_unit'), alpha = 0.027)
    
    # slice right states
    out = out.Hin
    out_obs = out_obs.Hin
    out_nat_obs = out_nat_obs.Hin

    # visualise
    dates_calibration = data.loc[slice(start_calibration, end_calibration), slice(None)].index.get_level_values('date').unique()
    dates_post_calibration = data.loc[slice(end_calibration+timedelta(days=1), end_simulation), slice(None)].index.get_level_values('date').unique()
    spatial_units = data.index.get_level_values('spatial_unit').unique()
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
                    ax.scatter(dates_calibration, data.loc[slice(start_calibration, end_calibration), spatial_units[j+counter]],
                                edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.6)
                    ax.scatter(dates_post_calibration, data.loc[slice(end_calibration+timedelta(days=1), end_simulation), spatial_units[j+counter]],
                                edgecolors='red', facecolors='white', marker='o', s=10, alpha=0.6)            
                    # plot model prediction
                    ax.plot(out.date, out.sel(spatial_unit=spatial_units[j+counter]).sum(dim='age_class').mean(dim='draws'), color='blue', linewidth=1)
                    ax.fill_between(out.date, out_obs.sel(spatial_unit=spatial_units[j+counter]).sum(dim='age_class').quantile(dim='draws', q=alpha/2), 
                                        out_obs.sel(spatial_unit=spatial_units[j+counter]).sum(dim='age_class').quantile(dim='draws', q=1-alpha/2), color='blue', alpha=0.2)      
                    # shade VOCs and vaccines
                    ax.axvspan('2021-02-01', end_simulation, color='black', alpha=0.1)
                    # set title
                    ax.set_title(spatial_units[j+counter])
                else:
                    # plot data
                    ax.scatter(dates_calibration, data.groupby(by='date').sum().loc[slice(start_calibration, end_calibration)],
                                edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.6)
                    ax.scatter(dates_post_calibration, data.groupby(by='date').sum().loc[slice(end_calibration+timedelta(days=1), end_simulation)],
                                edgecolors='red', facecolors='white', marker='o', s=10, alpha=0.6)            
                    # plot model prediction
                    ax.plot(out.date, out.sum(dim=['age_class', 'spatial_unit']).mean(dim='draws'), color='blue', linewidth=1)
                    ax.fill_between(out_nat_obs.date, out_nat_obs.sum(dim=['age_class']).quantile(dim='draws', q = alpha/2),
                                        out_nat_obs.sum(dim=['age_class']).quantile(dim='draws', q = 1-alpha/2), color='blue', alpha=0.2)
                    # shade VOCs and vaccines + text
                    ax.axvspan('2021-02-01', end_simulation, color='black', alpha=0.1)   
                    ax.text(0.72, 0.85, 'Vaccination\n   Variants', transform=ax.transAxes, fontsize=6)               
                    # set title
                    ax.set_title(country)
                # set maximum number of labels
                ax.xaxis.set_major_locator(MaxNLocator(5))
                # rotate labels
                for tick in ax.get_xticklabels():
                    tick.set_rotation(60)
                # limit xrange
                ax.set_xlim([start_calibration, end_simulation])
            else:
                fig.delaxes(ax)
        n_figs += 1
        counter += nrows*ncols
        plt.savefig(
            f'calibrate_together_{country}_part_{n_figs}.png', dpi=600)
        # plt.show()
        plt.close()