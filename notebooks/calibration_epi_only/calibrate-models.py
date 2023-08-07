from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, ll_negative_binomial, ll_poisson
from pySODM.optimization import pso, nelder_mead
from pyIEEM.data.data import get_hospitalisation_incidence
from pyIEEM.models.utils import initialize_model, aggregate_Brussels_Brabant_DataArray, dummy_aggregation
from datetime import date
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

abs_dir = os.path.dirname(__file__)

##########################
## change settings here ##
##########################

# settings calibration
start_calibration = '2020-03-07'
end_calibration = '2021-01-01'
processes = 6
max_iter = 200
multiplier_mcmc = 6
n_mcmc = 50
print_n = 5

# paths
identifier = 'poisson_enddate_20210201'
run_date = str(date.today())
fig_path = f''
samples_path = f''

# settings visualisation
nrows = 3
ncols = 4

#########################
## load model and data ##
#########################

# load data BE and SWE
data_BE = get_hospitalisation_incidence(
    'BE', aggregate_bxl_brabant=True).loc[slice(start_calibration, end_calibration)]
data_SWE = get_hospitalisation_incidence(
    'SWE').loc[slice(start_calibration, end_calibration)]

# load model BE and SWE
age_classes = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (
    35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 120)], closed='left')
model_BE = initialize_model('BE', age_classes, True, start_calibration)
model_SWE = initialize_model('SWE', age_classes, True, start_calibration)

# set up log likelihood function
models = [model_BE, model_SWE]
datasets = [data_BE, data_SWE]
dt = [data_BE, data_SWE]
states = ["Hin", "Hin"]
log_likelihood_fnc = [ll_negative_binomial, ll_negative_binomial]
aggregation_functions = [
    aggregate_Brussels_Brabant_DataArray, dummy_aggregation]
alpha = 0.027
log_likelihood_fnc_args = [len(data_BE.index.get_level_values('spatial_unit').unique())*[alpha,],
                           len(data_SWE.index.get_level_values('spatial_unit').unique())*[alpha,]]                  
pars = ['tau', 'ypsilon_eff', 'phi_eff', 'phi_work', 'phi_leisure']
bounds = ((1, 100), (0, 100), (0, 100), (0, 100), (0, 100))
labels = [r'$\tau$', r'$\upsilon_{eff}$', r'$\phi_{eff}$', r'$\phi_{work}$', r'$\phi_{leisure}$']
weights = [1/len(data_BE), 1/len(data_SWE)]
objective_function = log_posterior_probability(models, pars, bounds, datasets, states, log_likelihood_fnc, log_likelihood_fnc_args,
                                               start_sim=start_calibration, aggregation_function=aggregation_functions, labels=labels)

if __name__ == '__main__':

    ####################
    ## NM calibration ##
    ####################
    
    # starting point
    theta = [2.40461891e+01, 4.82539554e-01, 5.63209689e-02, 1.18561822e-02, 3.24334640e-01] # ll: 13130; calibration begin Jan 2020
    theta = [1.83288315e+01, 4.84588154e-01, 5.35750249e-02, 1.54308721e-02, 2.79468411e-01] ## ll: 12370; calibration begin Jan 2020
    theta = nelder_mead.optimize(objective_function, np.array(theta), len(bounds)*[1,], processes=processes, max_iter=max_iter)[0]

    # visualisation
    for i, country in enumerate(['BE', 'SWE']):

        # set right model and data
        model = models[i]
        data = dt[i]

        # set optimal parameters
        for k, par in enumerate(pars):
            model.parameters.update({par: theta[k]})

        # simulate model
        out = model.sim([start_calibration, end_calibration])

        # aggregate model
        out = aggregation_functions[i](out.Hin)

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
                        ax.plot(out.date, out.sum(dim='age_class').sel(
                            spatial_unit=spatial_units[j+counter]), color='red')
                        # set title
                        ax.set_title(spatial_units[j+counter])
                    else:
                        # plot data
                        ax.scatter(dates, data.groupby(by='date').sum().loc[slice(
                            None)], edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.8)
                        # plot model prediction
                        ax.plot(out.date, out.sum(
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
            plt.savefig(
                f'calibrate_together_{country}_part_{n_figs}.png', dpi=600)
            #plt.show()
            plt.close()

    import sys
    sys.exit()

    ##########
    ## MCMC ##
    ##########

    ndim, nwalkers, pos = perturbate_theta(theta, len(
        pars)*[0.05,], multiplier=multiplier_mcmc, bounds=bounds, verbose=False)

    # Write settings to a .txt
    settings = {'start_calibration': start_calibration, 'end_calibration': end_calibration, 'n_chains': nwalkers,
                'dispersion': alpha, 'labels': labels, 'starting_estimate': list(theta)}

    print(
        f'Using {processes} cores for {ndim} parameters, in {nwalkers} chains.\n')
    sys.stdout.flush()

    # Setup sampler
    sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function, print_n=print_n, backend=None, processes=processes,
                                  samples_path=samples_path, fig_path=fig_path, progress=True, settings_dict=settings)

    # Sample up to 40*n_mcmc more
    import emcee
    for i in range(40):
        backend = emcee.backends.HDFBackend(os.path.join(
            os.getcwd(), samples_path+identifier+'_BACKEND_'+run_date+'.hdf5'))
        sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function, print_n=print_n, backend=backend, processes=processes,
                                      samples_path=samples_path, fig_path=fig_path, progress=True, settings_dict=settings)

    #####################
    ## Process results ##
    #####################

    # Generate a sample dictionary
    samples_dict = emcee_sampler_to_dictionary(
        sampler, discard=1, identifier=identifier, samples_path=samples_path, settings=settings)
    # Save samples dictionary to json
    import json
    with open(samples_path+str(identifier)+'_SAMPLES_'+run_date+'.json', 'w') as fp:
        json.dump(samples_dict, fp)

    print('DONE!')
    print('SAMPLES DICTIONARY SAVED IN '+'"'+samples_path +
          str(identifier)+'_SAMPLES_'+run_date+'.json'+'"')
    print('-----------------------------------------------------------------------------------------------------------------------------------\n')
