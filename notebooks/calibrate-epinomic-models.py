from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, ll_negative_binomial, ll_gaussian, ll_poisson, log_prior_normal_L2
from pySODM.optimization import pso, nelder_mead
from pyIEEM.data.data import get_hospitalisation_incidence, get_economic_data
from pyIEEM.models.utils import initialize_epinomic_model, aggregate_Brussels_Brabant_DataArray, dummy_aggregation
from datetime import date
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
end_calibration_epi = '2021-01-01'
end_calibration_eco = '2020-12-01'
processes = 36
max_iter = 100
multiplier_mcmc = 3
n_mcmc = 100
print_n = 5

# paths
identifier = 'calibration_ICU'
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
data_BE_eco_GDP = get_economic_data('GDP', 'BE', relative=False).loc[slice(start_calibration, end_calibration_eco)]
data_SWE_eco_GDP = get_economic_data('GDP', 'SWE', relative=False).loc[slice(start_calibration, end_calibration_eco)]
data_BE_eco_employment = get_economic_data('employment', 'BE', relative=False).loc[slice(start_calibration, end_calibration_eco)]
data_SWE_eco_employment = get_economic_data('employment', 'SWE', relative=False).loc[slice(start_calibration, end_calibration_eco)]

# load model BE and SWE
age_classes = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (
    35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 120)], closed='left')
model_BE = initialize_epinomic_model('BE', age_classes, True, start_calibration, prodfunc='half_critical')
model_SWE = initialize_epinomic_model('SWE', age_classes, True, start_calibration, prodfunc='half_critical')

# set up log likelihood function
models = [model_BE, model_SWE, model_BE, model_SWE, model_BE, model_SWE]
datasets = [data_epi_BE, data_epi_SWE, data_BE_eco_GDP, data_SWE_eco_GDP, data_BE_eco_employment, data_SWE_eco_employment]
dt_epi = [data_epi_BE, data_epi_SWE]
dt_eco_GDP = [data_BE_eco_GDP, data_SWE_eco_GDP]
dt_eco_employment = [data_BE_eco_GDP, data_SWE_eco_GDP]

states = ["Hin", "Hin", "x", "x", "l", "l"]
log_likelihood_fnc = [ll_negative_binomial, ll_poisson, ll_gaussian, ll_gaussian, ll_gaussian, ll_gaussian]
aggregation_functions = [aggregate_Brussels_Brabant_DataArray, dummy_aggregation, dummy_aggregation, dummy_aggregation, dummy_aggregation, dummy_aggregation]
alpha = 0.036 # national overdispersion BE
log_likelihood_fnc_args = [[0.05, 0.039, 0.024, 0.061, 0.068, 0.014, 0.10, 0.03, 0.07], [], 0.02, 0.02, 0.02, 0.02]
weights = [1/len(data_epi_BE), 1/len(data_epi_SWE), 1/len(data_BE_eco_GDP), 1/len(data_SWE_eco_GDP), 1/len(data_BE_eco_employment), 1/len(data_SWE_eco_employment)]

# parameter properties
pars = ['nu', 'xi_eff', 'pi_eff', 'pi_work', 'pi_leisure', 'mu', 'amplitude_BE', 'peak_shift_BE', 'amplitude_SWE', 'peak_shift_SWE', 'iota_H', 'iota_F']
bounds = ((1, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0,2), (0,0.40), (-31,31), (0,0.40), (-31,31), (1,31),(1,31))
labels = [r'$\nu$', r'$\xi_{eff}$', r'$\pi_{eff}$', r'$\pi_{work}$', r'$\pi_{leisure}$', r'$\mu$', r'$A_{BE}$',  r'$\Delta A_{BE}$', r'$A_{SWE}$',  r'$\Delta A_{SWE}$', r'$\iota_H$', r'$\iota_F$']
# reguralised prior probabilities (this does require some feeling)
# all prior probabilities parameters were set so that a score of roughly -250 (on a total of -16000) is added to the posterior probability when the parameter leaves the range I would expect them to fall in
# this will have to be balanced by trial-and-error
log_prior_prob_fnc=[log_prior_normal_L2, log_prior_normal_L2, log_prior_normal_L2, log_prior_normal_L2, log_prior_normal_L2, log_prior_normal_L2, log_prior_normal_L2, log_prior_normal_L2,
                        log_prior_normal_L2, log_prior_normal_L2, log_prior_normal_L2, log_prior_normal_L2]

theta = [24, 0.45, 0.07, 0.035, 0.06, 1, 0.20, -7, 0.20, 14, 7, 7] # where are my parameters? 
mu_list = [24, 0.45, 0.07, 0.035, 0.06, 1, 0.20, 0, 0.20, 0, 7, 7] # where do I expect the parameters to be?
sigma_list = [2, 0.03, 0.005, 0.0035, 0.006, 0.2, 0.02, 7, 0.03, 7, 2, 2] # How much noise do I expect there to be on the parameter value?
l_list = [8, 10, 25, 25, 15, 10, 20, 16, 20, 14, 10, 10] # How strong are my beliefs?
log_prior_prob_fnc_args=[]
for mu,sigma,l in zip(mu_list,sigma_list, l_list):
    log_prior_prob_fnc_args += [(mu, sigma, l),]

# construct log likelihood
objective_function = log_posterior_probability(models, pars, bounds, datasets, states, log_likelihood_fnc, log_likelihood_fnc_args,
                                                log_prior_prob_fnc=log_prior_prob_fnc, log_prior_prob_fnc_args=log_prior_prob_fnc_args,
                                                start_sim=start_calibration, aggregation_function=aggregation_functions, labels=labels)

if __name__ == '__main__':

    ####################
    ## NM calibration ##
    ####################
    
    # starting point
    #theta = [22, 0.45, 0.07, 0.025, 0.06]
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
    #     fig,ax=plt.subplots(ncols=2)
    #     # gdp
    #     ax[0].scatter([data_BE_eco_GDP, data_SWE_eco_GDP][i].index.get_level_values('date').unique(), [data_BE_eco_GDP, data_SWE_eco_GDP][i],
    #                 edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.8)
    #     ax[0].plot(out.date, out.x.sum(dim='NACE64'), color='red')
    #     # employment
    #     ax[1].scatter([data_BE_eco_employment, data_SWE_eco_employment][i].index.get_level_values('date').unique(), [data_BE_eco_employment, data_SWE_eco_employment][i],
    #                 edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.8)
    #     ax[1].plot(out.date, out.l.sum(dim='NACE64'), color='red')
    #     plt.savefig(
    #             f'epinomic_eco_{country}.png', dpi=600)
    #     #plt.show()
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
    #         #plt.show()
    #         plt.close()

    # sys.exit()

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
