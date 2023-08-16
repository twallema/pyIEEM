from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, ll_negative_binomial, ll_gaussian
from pySODM.optimization import pso, nelder_mead
from pyIEEM.data.data import get_hospitalisation_incidence, get_economic_data
from pyIEEM.models.utils import initialize_epidemic_model, initialize_epinomic_model, aggregate_Brussels_Brabant_DataArray, dummy_aggregation
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
end_calibration_epi = '2021-02-01'
end_calibration_eco = '2021-01-01'
processes = 1
max_iter = 200
multiplier_mcmc = 3
n_mcmc = 50
print_n = 5

# paths
identifier = 'enddate_20210201'
run_date = str(date.today())
fig_path = f''
samples_path = f''

# settings visualisation
nrows = 3
ncols = 4

#########################
## load model and data ##
#########################

# load epidemiological data BE and SWE
data_epi_BE = get_hospitalisation_incidence('BE', aggregate_bxl_brabant=True).loc[slice(start_calibration, end_calibration_epi)]
data_epi_SWE = get_hospitalisation_incidence('SWE').loc[slice(start_calibration, end_calibration_epi)]

# load economic data BE and SWE
data_eco_BE = get_economic_data('GDP', 'BE').loc[slice(start_calibration, end_calibration_eco)]
data_eco_SWE = get_economic_data('GDP', 'SWE').loc[slice(start_calibration, end_calibration_eco)]

# normalise Swedish GDP data to one
data_eco_SWE *= 29880/data_eco_SWE.iloc[0]

# load model BE and SWE
age_classes = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (
    35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 120)], closed='left')
model_BE = initialize_epinomic_model('BE', age_classes, True, start_calibration, prodfunc='half_critical')
model_SWE = initialize_epinomic_model('SWE', age_classes, True, start_calibration, prodfunc='half_critical')

# set up log likelihood function
models = [model_BE, model_SWE, model_BE, model_SWE]
datasets = [data_epi_BE, data_epi_SWE, data_eco_BE, data_eco_SWE]
dt_epi = [data_epi_BE, data_epi_SWE]
dt_eco = [data_eco_BE, data_eco_SWE]
states = ["Hin", "Hin", "x", "x"]
log_likelihood_fnc = [ll_negative_binomial, ll_negative_binomial, ll_gaussian, ll_gaussian]
aggregation_functions = [
    aggregate_Brussels_Brabant_DataArray, dummy_aggregation, dummy_aggregation, dummy_aggregation]
alpha = 0.027
log_likelihood_fnc_args = [len(data_epi_BE.index.get_level_values('spatial_unit').unique())*[alpha,],
                           len(data_epi_SWE.index.get_level_values('spatial_unit').unique())*[alpha,], 0.01, 0.01]

pars = ['nu', 'xi_eff', 'pi_eff', 'pi_work', 'pi_leisure', 'amplitude']
bounds = ((1, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0,0.40))
labels = [r'$\nu$', r'$\xi_{eff}$', r'$\pi_{eff}$', r'$\pi_{work}$', r'$\pi_{leisure}$', r'$A$']
weights = [1/len(data_epi_BE), 1/len(data_epi_SWE), 1/len(data_eco_BE), 1/len(data_eco_SWE)]
objective_function = log_posterior_probability(models, pars, bounds, datasets, states, log_likelihood_fnc, log_likelihood_fnc_args,
                                               start_sim=start_calibration, aggregation_function=aggregation_functions, labels=labels)

if __name__ == '__main__':

    ####################
    ## NM calibration ##
    ####################
    
    # starting point
    theta = [20, 4.82539554e-01, 5.63209689e-02, 0.03, 0.10, 0.20]
    #theta = nelder_mead.optimize(objective_function, np.array(theta), len(bounds)*[1,], processes=processes, max_iter=max_iter)[0]

    # visualisation epi data
    # for i, country in enumerate(['BE', 'SWE']):

    #     # set right model and data
    #     model = models[i]
    #     data = dt_epi[i]

    #     # set optimal parameters
    #     for k, par in enumerate(pars):
    #         model.parameters.update({par: theta[k]})

    #     # simulate model
    #     out = model.sim([start_calibration, end_calibration_epi])

    #     # visualise eco
    #     fig,ax=plt.subplots()
    #     ax.scatter(dt_eco[i].index.get_level_values('date').unique(), dt_eco[i],
    #                 edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.8)
    #     ax.plot(out.date, out.x.sum(dim='NACE64'), color='red')
    #     plt.savefig(
    #             f'epinomic_eco_{country}.png', dpi=600)
    #     plt.show()
    #     plt.close()

    #     # aggregate model
    #     out = aggregation_functions[i](out.Hin)

    #     # visualise epi
    #     dates = data.index.get_level_values('date').unique()
    #     spatial_units = data.index.get_level_values('spatial_unit').unique()
    #     n_figs = 0
    #     counter = 0
    #     while counter <= len(spatial_units):
    #         fig, axes = plt.subplots(
    #             nrows=nrows, ncols=ncols, figsize=(11.7, 8.3), sharex=True)
    #         axes = axes.flatten()
    #         for j, ax in enumerate(axes):
    #             if j+counter <= len(spatial_units):
    #                 if j + counter < len(spatial_units):
    #                     # plot data
    #                     ax.scatter(dates, data.loc[slice(None), spatial_units[j+counter]],
    #                                edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.8)
    #                     # plot model prediction
    #                     ax.plot(out.date, out.sum(dim='age_class').sel(
    #                         spatial_unit=spatial_units[j+counter]), color='red')
    #                     # set title
    #                     ax.set_title(spatial_units[j+counter])
    #                 else:
    #                     # plot data
    #                     ax.scatter(dates, data.groupby(by='date').sum().loc[slice(
    #                         None)], edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.8)
    #                     # plot model prediction
    #                     ax.plot(out.date, out.sum(
    #                         dim=['age_class', 'spatial_unit']), color='red')
    #                     # set title
    #                     ax.set_title(country)
    #                 # set maximum number of labels
    #                 ax.xaxis.set_major_locator(MaxNLocator(5))
    #                 # rotate labels
    #                 for tick in ax.get_xticklabels():
    #                     tick.set_rotation(60)
    #             else:
    #                 fig.delaxes(ax)
    #         n_figs += 1
    #         counter += nrows*ncols
    #         plt.savefig(
    #             f'epinomic_epi_{country}_part_{n_figs}.png', dpi=600)
    #         plt.show()
    #         plt.close()

    ##########
    ## MCMC ##
    ##########

    ndim, nwalkers, pos = perturbate_theta(theta, len(
        pars)*[0.05,], multiplier=multiplier_mcmc, bounds=bounds, verbose=False)

    # Write settings to a .txt
    settings = {'start_calibration': start_calibration, 'end_calibration_epi': end_calibration_epi, 'end_calibration_eco': end_calibration_eco, 'n_chains': nwalkers,
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
