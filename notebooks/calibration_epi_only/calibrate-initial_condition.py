from pySODM.optimization import pso, nelder_mead
from pyIEEM.models.utils import initialize_epidemic_model
from pyIEEM.data.data import get_hospitalisation_incidence
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from scipy.special import gammaln
import pandas as pd
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"

##########################
## change settings here ##
##########################

# settings calibration
start_calibration = '2020-02-01'
processes = 18
max_iter = 200

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
    N = np.sum(N_other, axis=0) + np.sum(N_work, axis=0)
    # pre-allocate output
    output = np.zeros(initial_condition.shape, dtype=float)
    # loop over spatial patches
    for j, inf in enumerate(infected):
        # distribute over age groups
        inf = inf * (N[:, 0]/sum(N[:, 0]))
        # determine index of age group to drop infected in
        # i = np.random.choice(len(N[:, 0]), p=N[:, 0]/sum(N[:, 0]))
        # assign to output
        output[:, j] = inf
    return output

# custom SSE function
def poisson_ll(theta, data, model, start_calibration, end_calibration):
    """
    A custom Poisson log-likelihood function
    """
    # change initial condition
    model.initial_states['E'] = update_initial_condition(
        theta, model.initial_states['E'], model.parameters['N']['other'], model.parameters['N']['work'])
    # simulate model
    out = model.sim([start_calibration, end_calibration])
    # slice right data out of dataframe
    data = data.loc[slice(start_calibration, end_calibration)]
    # interpolate model output to dates in data
    out = out.interp({'date': data.index.get_level_values(
        'date').unique().values}, method="linear")
    # extract hospitalisations
    ymodel = out.Hin.sum(dim='age_class').sel(
        spatial_unit=data.index.get_level_values('spatial_unit').unique().values).values + 1
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
        end_calibration = '2020-05-16'
    else:
        end_calibration = '2020-04-02'
    data = data.loc[slice(start_calibration, end_calibration)]

    # setup model
    age_classes = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (
        35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 120)], closed='left')
    model = initialize_epidemic_model(country, age_classes, True, start_calibration)

    # disable any awareness triggering (discontinued)
    # model.parameters.update({'l': 21, 'mu': 1, 'nu': 24, 'xi_work': 100, 'xi_eff': 100, 'xi_leisure': 100,
    #                     'pi_work': 1, 'pi_eff': 1, 'pi_leisure': 1})

    # use good parameter values found during an earlier calibration (this is an iterative way of finding a suitable initial condition)
    # define all relevant parameters of the social contact function TDPF here
    model.parameters.update({'l': 5, 'mu': 1, 'nu': 24, 'xi_work': 5, 'xi_eff': 0.50, 'xi_leisure': 5,
                        'pi_work': 0.02, 'pi_eff': 0.06, 'pi_leisure': 0.30})
    
    # set a good parameter estimate
    pars = ['nu', 'xi_eff', 'pi_eff', 'pi_work', 'pi_leisure']
    theta = [22, 0.45, 0.07, 0.025, 0.06]
    for par,t in zip(pars,theta):
        model.parameters.update({par: t})

    # compute number of spatial patches
    G = model.initial_states['E'].shape[1]

    # method used: started from an initial guess, did some manual tweaks to the output, gave that back to the NM optimizer, etc.
    if country == 'SWE':
        # data is quite consistent with one infected in Stockholm --> start NM from here
        theta = 0.22*np.array([0, 0, 0.02, 0, 0, 0, 0, 0, 0, 0, 0.03, 0.12, 0.01, 0.15, 1, 0.05, 0, 0, 0, 0, 0.02]) + 1e-9
    else:
        theta = 0.16*np.array([0.85, 0, 0, 3.25, 1.75, 2.50, 0.25, 0, 1.50, 0.25, 0.50]) + 1e-9 # "best" fit
        theta = [1.60090552e-01, 7.52869047e-10, 1.50237870e-09, 4.69319925e-01,
                    3.00200083e-01, 3.00129689e-01, 6.00489179e-02, 2.00130982e-09,
                    2.55561357e-01, 4.91375446e-05, 9.30272369e-02]

    # nelder-mead minimization
    #theta = nelder_mead.optimize(poisson_ll, np.array(theta), 1*np.ones(len(theta)), bounds=G*[(0, 100)],
    #                             args=(data, model, start_calibration, end_calibration), processes=processes, max_iter=max_iter)[0]

    # set found initial condition
    model.initial_states['E'] = update_initial_condition(
        theta, model.initial_states['E'], model.parameters['N']['other'], model.parameters['N']['work'])

    # simulate
    out = model.sim([start_calibration, end_calibration])

    # visualize data and model prediction
    dates = data.index.get_level_values('date').unique()
    spatial_units = data.index.get_level_values('spatial_unit').unique()
    n_figs = 0
    counter = 0
    while counter <= len(spatial_units):
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(11.7, 8.3), sharex=True)
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i+counter <= len(spatial_units):
                if i + counter < len(spatial_units):
                    # plot data
                    ax.scatter(dates, data.loc[slice(None), spatial_units[i+counter]],
                               edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.8)
                    # plot model prediction
                    ax.plot(out.date, out.Hin.sum(dim='age_class').sel(
                        spatial_unit=spatial_units[i+counter]), color='red')
                    # set title
                    ax.set_title(spatial_units[i+counter])
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
        plt.savefig(f'initial_condition_{country}_part_{n_figs}.png', dpi=600)
        plt.show()
        plt.close()

    #################
    ## save result ##
    #################

    abs_dir = os.path.dirname(__file__)
    out.coords.update({'age_class': range(len(out.coords['age_class']))})
    out.to_netcdf(os.path.join(
        abs_dir, f'../../data/interim/epi/initial_condition/{country}_INITIAL_CONDITION.nc'))
