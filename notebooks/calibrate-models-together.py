import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import date
from pyIEEM.models.utils import initialize_model
from pyIEEM.data.data import get_hospitalisation_incidence
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.objective_functions import log_posterior_probability, ll_negative_binomial
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary

abs_dir = os.path.dirname(__file__)

##########################
## change settings here ##
##########################

# settings calibration
#start_calibration = ['2020-03-13','2020-02-27'] # moment BE and SWE pass 4 hospitalised COVID-19 patients per 100K inhabitants
#start_calibration = ['2020-03-14','2020-02-28'] # moment BE and SWE pass 5 hospitalised COVID-19 patients per 100K inhabitants
start_calibration = ['2020-03-15','2020-02-29'] # moment BE and SWE pass 6 hospitalised COVID-19 patients per 100K inhabitants
#start_calibration = ['2020-03-16','2020-03-01'] # moment BE and SWE pass 7 hospitalised COVID-19 patients per 100K inhabitants

end_calibration = '2020-10-01'
processes = 6
max_iter = 50
multiplier_mcmc = 6
n_mcmc = 200
print_n = 10

# paths
identifier = 'test'
run_date = str(date.today())
fig_path = f''
samples_path = f''

# settings visualisation
nrows = 3
ncols = 4
    
#########################
## load model and data ##
#########################

## load data BE and SWE
data_BE = get_hospitalisation_incidence('BE').loc[slice(start_calibration[0], end_calibration)]
data_SWE = get_hospitalisation_incidence('SWE').loc[slice(start_calibration[1], end_calibration)]


## load model BE and SWE
age_classes = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (
        35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 120)], closed='left')
model_BE = initialize_model('BE', age_classes, True, start_calibration[0])
model_SWE = initialize_model('SWE', age_classes, True, start_calibration[1])

## set up log likelihood function
models = [model_BE, model_SWE]
datasets = [data_BE, data_SWE]
states = ["Hin", "Hin"]
log_likelihood_fnc = [ll_negative_binomial, ll_negative_binomial] 
alpha = 0.03
log_likelihood_fnc_args = [len(data_BE.index.get_level_values('spatial_unit').unique())*[alpha,],
                            len(data_SWE.index.get_level_values('spatial_unit').unique())*[alpha,]]
pars = ['tau', 'ypsilon_eff', 'phi_eff', 'ypsilon_work', 'phi_work', 'amplitude']
bounds=((5,100),(0,100),(0,100),(5,100),(0,100),(0,1))
labels = [r'$\tau$', r'$\upsilon_{eff}$', r'$\phi_{eff}$', r'$\upsilon_{work}$', r'$\phi_{work}$', r'$A$']
weights = [1/len(data_BE), 1/len(data_SWE)]
objective_function = log_posterior_probability(models, pars, bounds, datasets, states, log_likelihood_fnc,
                                                log_likelihood_fnc_args, start_sim=start_calibration, labels=labels)

####################
## NM calibration ##
####################

theta = [7,  0.60,  0.03, 10, 0.20,  0.40] ## ll: 1.209e+04 
theta = [10.63413066,  0.64575617,  0.02872299, 15.19161523,  0.07914095,  0.47191358]

theta = pso.optimize(objective_function, bounds, swarmsize=5*processes, max_iter=max_iter, processes=processes, debug=True)[0]

#theta = nelder_mead.optimize(objective_function, np.array(theta), len(bounds)*[0.50,], processes=processes, max_iter=max_iter)[0]

## visualisation
for i, country in enumerate(['BE', 'SWE']):

    # set right model and data
    model = models[i]
    data = datasets[i]
    start_sim = start_calibration[i]

    # set optimal parameters
    for k, par in enumerate(pars):
        model.parameters.update({par: theta[k]})

    # simulate model
    out = model.sim([start_sim, end_calibration])

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
        plt.savefig(f'calibrate_together_{country}_part_{n_figs}.png', dpi=600)
        #plt.show()
        plt.close()

##########
## MCMC ##
##########

if __name__ == '__main__':
        
    ndim, nwalkers, pos = perturbate_theta(theta, len(pars)*[0.25,], multiplier=multiplier_mcmc, bounds=bounds, verbose=False)

    # Write settings to a .txt
    settings={'start_calibration': start_calibration, 'end_calibration': end_calibration, 'n_chains': nwalkers,
                'dispersion': alpha, 'labels': labels, 'starting_estimate': theta}


    print(f'Using {processes} cores for {ndim} parameters, in {nwalkers} chains.\n')
    sys.stdout.flush()

    # Setup sampler
    sampler = run_EnsembleSampler(pos, 100, identifier, objective_function,print_n=print_n, backend=None, processes=processes,
                                    samples_path=samples_path, fig_path=fig_path, progress=True, settings_dict=settings) 

    # Sample up to 40*n_mcmc more
    import emcee
    for i in range(40):
        backend = emcee.backends.HDFBackend(os.path.join(os.getcwd(),samples_path+identifier+'_BACKEND_'+run_date+'.hdf5'))
        sampler = run_EnsembleSampler(pos, 100, identifier, objective_function,print_n=print_n, backend=backend, processes=processes,
                                        samples_path=samples_path, fig_path=fig_path, progress=True, settings_dict=settings)   

    #####################
    ## Process results ##
    #####################

    # Generate a sample dictionary
    samples_dict = emcee_sampler_to_dictionary(sampler, discard=1, identifier=identifier, samples_path=samples_path, settings=settings)
    # Save samples dictionary to json
    import json
    with open(samples_path+str(identifier)+'_SAMPLES_'+run_date+'.json', 'w') as fp:
        json.dump(samples_dict, fp)

    print('DONE!')
    print('SAMPLES DICTIONARY SAVED IN '+'"'+samples_path+str(identifier)+'_SAMPLES_'+run_date+'.json'+'"')
    print('-----------------------------------------------------------------------------------------------------------------------------------\n')
