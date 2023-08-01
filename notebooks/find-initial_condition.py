import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from scipy.special import gammaln
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pyIEEM.data.data import get_hospitalisation_incidence
from pyIEEM.models.utils import initialize_model
from pySODM.optimization import pso, nelder_mead

##########################
## change settings here ##
##########################

# settings calibration
country = 'SWE'
start_calibration = '2020-02-01'
processes = 6
max_iter = 100

# settings visualisation
nrows = 3
ncols = 4

######################
## helper functions ##
######################

# helper function to adjust initial condition
def update_initial_condition(infected, initial_condition, N_other, N_work):
    assert len(infected) == initial_condition.shape[1]
    # compute number of contacts per age group per spatial patch (N x G)
    N = np.sum(N_other,axis=0) + np.sum(N_work, axis=0)
    # pre-allocate output
    output = np.zeros(initial_condition.shape, dtype=float)
    # loop over spatial patches
    for j,inf in enumerate(infected):
        # distribute over age groups
        inf = inf * (N[:, 0]/sum(N[:, 0]))
        # determine index of age group to drop infected in
        #i = np.random.choice(len(N[:, 0]), p=N[:, 0]/sum(N[:, 0]))
        # assign to output
        output[:,j] = inf
    return output

# custom SSE function
def poisson_ll(theta, data, model, start_calibration, end_calibration):
    """
    A custom Poisson log-likelihood function
    """
    # change initial condition
    model.initial_states['E'] = update_initial_condition(theta, model.initial_states['E'], model.parameters['N']['other'], model.parameters['N']['work'])
    # simulate model
    out = model.sim([start_calibration, end_calibration])
    # slice right data out of dataframe
    data = data.loc[slice(start_calibration, end_calibration)]
    # interpolate model output to dates in data
    out = out.interp({'date': data.index.get_level_values('date').unique().values}, method="linear")
    # extract hospitalisations
    ymodel = out.Hin.sum(dim='age_class').sel(spatial_unit=data.index.get_level_values('spatial_unit').unique().values).values + 1
    # compute poisson likelihood
    ydata = data.values.reshape(ymodel.shape) + 1
    return - np.sum(ymodel) + np.sum(ydata*np.log(ymodel)) - np.sum(gammaln(ydata))

#################
## Calibration ##
#################

for country in ['BE', 'SWE']:

    # get data
    data = get_hospitalisation_incidence(country)

    # slice data until calibration end
    if country == 'SWE':
        end_calibration = '2020-03-10'
    else:
        end_calibration = '2020-03-22'
    data = data.loc[slice(start_calibration, end_calibration)]

    # setup model
    age_classes = pd.IntervalIndex.from_tuples([(0,5),(5,10),(10,15),(15,20),(20,25),(25,30),(30,35),(35,40),(40,45),(45,50),(50,55),(55,60),(60,65),(65,70),(70,75),(75,80),(80,120)], closed='left')
    model = initialize_model(country, age_classes, True, start_calibration)

    # compute number of spatial patches
    G = model.initial_states['E'].shape[1]

    # method used: started from an initial guess, did some manual tweaks to the output, gave that back to the NM optimizer, etc.
    if country == 'SWE':       
        theta = np.array([1e-9, 0.188, 0.165, 0.049, 0.165, 0.065, 0.762, 1e-9, 0.00695, 0.0416, 0.274, 0.850,
                            0.249, 0.296, 7.93, 0.0807, 0.00276, 0.00417, 0.00185, 0.095, 0.265]) # ll: 210
    else:
        theta = np.array([0.82, 0.0, 0.031, 0.0, 0.97, 0.87, 1.21, 1.09, 0.0, 0.0, 0.45]) # ll: 270

    # nelder-mead minimization
    #theta = nelder_mead.optimize(SSE, np.array(theta), 1*np.ones(len(theta)), bounds=G*[(0, 100)],
    #                                args=(data, model, start_calibration, end_calibration), processes=processes, max_iter=max_iter)[0]

    # set found initial condition
    model.initial_states['E'] = update_initial_condition(theta, model.initial_states['E'], model.parameters['N']['other'], model.parameters['N']['work'])

    # simulate
    out = model.sim([start_calibration, end_calibration])

    # visualize data and model prediction
    dates = data.index.get_level_values('date').unique()
    spatial_units = data.index.get_level_values('spatial_unit').unique()
    n_figs = 0
    counter = 0
    while counter <= len(spatial_units):
        fig,axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11.7, 8.3), sharex=True)
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i+counter <= len(spatial_units):
                if i+ counter < len(spatial_units):
                    # plot data
                    ax.scatter(dates, data.loc[slice(None), spatial_units[i+counter]], edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.8)
                    # plot model prediction
                    ax.plot(out.date, out.Hin.sum(dim='age_class').sel(spatial_unit=spatial_units[i+counter]), color='red')
                    # set title
                    ax.set_title(spatial_units[i+counter])                
                else:
                    # plot data
                    ax.scatter(dates, data.groupby(by='date').sum().loc[slice(None)], edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.8)
                    # plot model prediction
                    ax.plot(out.date, out.Hin.sum(dim=['age_class', 'spatial_unit']), color='red')
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
        plt.savefig(f'initial_condition_{country}_part_{n_figs}.pdf')
        #plt.show()
        plt.close()

    #################
    ## save result ##
    #################

    abs_dir = os.path.dirname(__file__)
    out.coords.update({'age_class': range(len(out.coords['age_class']))})
    out.to_netcdf(os.path.join(abs_dir, f'../data/interim/epi/initial_condition/{country}_INITIAL_CONDITION.nc'))