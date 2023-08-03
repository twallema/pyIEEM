import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from pyIEEM.models.utils import initialize_model
from pyIEEM.data.data import get_hospitalisation_incidence
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.objective_functions import log_posterior_probability, ll_negative_binomial

##########################
## change settings here ##
##########################

# settings calibration
start_calibration = '2020-02-21'
end_calibration = '2021-01-01'
processes = 6
max_iter = 20
# settings visualisation
nrows = 3
ncols = 4

for i, country in enumerate(['SWE', 'BE']):

    ## load data BE and SWE
    data = get_hospitalisation_incidence(country).loc[slice(start_calibration, end_calibration)]

    ## load model BE and SWE
    age_classes = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (
            35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 120)], closed='left')
    model = initialize_model(country, age_classes, True, start_calibration)
    G = model.initial_states['E'].shape[1]

    ## set up log likelihood function
    alpha = len(data.index.get_level_values('spatial_unit').unique())*[0.03,] # negative-binomial model
    states = ["Hin",]
    log_likelihood_fnc = [ll_negative_binomial,] 
    log_likelihood_fnc_args = [alpha,]
    pars = ['tau', 'ypsilon_eff', 'phi_eff', 'amplitude']
    bounds=((1,100),(0,100),(0,100), (0,0.50))
    labels = ['$\\tau$', '$\\ypsilon$', '$\\phi$', '$A$']
    objective_function = log_posterior_probability(model, pars, bounds, [data, ], states, log_likelihood_fnc, log_likelihood_fnc_args, labels=labels)

    ## NM calibration
    theta = [41.38326874,  2.08266602,  0.16777618,  0.2348111] # BE
    theta = [31,  3,  0.2,  0.001] # SWE
    theta = nelder_mead.optimize(objective_function, np.array(theta), len(bounds)*[0.20,], processes=processes, max_iter=max_iter)[0]

    ## visualisation

    # set optimal parameters
    for k, par in enumerate(pars):
        model.parameters.update({par: theta[k]})

    # simulate model
    out = model.sim([start_calibration, end_calibration])

    # visualise
    dates = data.index.get_level_values('date').unique()
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
                    ax.scatter(dates, data.loc[slice(None), spatial_units[j+counter]],
                                edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.8)
                    # plot model prediction
                    ax.plot(out.date, out.Hin.sum(dim='age_class').sel(
                        spatial_unit=spatial_units[j+counter]), color='red')
                    # set title
                    ax.set_title(spatial_units[j+counter])
                else:
                    # plot data
                    ax.scatter(dates, data.groupby(by='date').sum().loc[slice(
                        None)], edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.8)
                    # plot model prediction
                    ax.plot(out.date, out.Hin.sum(
                        dim=['age_class', 'spatial_unit']), color='red')
                    # set title
                    ax.set_title(country)
                # set maximum number of labels
                ax.xaxis.set_major_locator(MaxNLocator(5))
                # rotate labels
                for tick in ax.get_xticklabels():
                    tick.set_rotation(60)
            else:
                fig.delaxes(ax)
        n_figs += 1
        counter += nrows*ncols
        plt.savefig(f'calibrate_seperately_{country}_part_{n_figs}.png', dpi=600)
        plt.show()
        plt.close()
