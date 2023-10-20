import os
import pandas as pd
import matplotlib.pyplot as plt

# define colorscale
cmap = {"orange" : "#E69F00", "light_blue" : "#56B4E9",
        "green" : "#009E73", "yellow" : "#F0E442",
        "blue" : "#0072B2", "red" : "#D55E00",
        "pink" : "#CC79A7", "black" : "#000000"}

##############
## settings ##
##############

parameter_name = 'nu'
parameter_name_fancy = 'Memory\nhalf- life'
parameter_unit = 'd.'
IC_ratio = 0.162
population = [11.6e6, 10.4e6]
IC_beds_nominal = [1000, 600]
IC_beds_extended = [2000, 1000]
location_IC_annotation = 175
IC_multipliers = [IC_beds_nominal[1]/IC_beds_nominal[i] for i in range(len(IC_beds_nominal))]
states = ['Ih', 'x', 'l']
countries = ['BE', 'SWE']
country_names = ['Belgium', 'Sweden']
colors = [cmap['blue'], cmap['green'], cmap['red'], cmap['black']]
ylabels = ['IC load (beds)', 'Gross\nagg. output (%)', 'Labor\ncompensation (%)']
ylimits = [[[0, 7.75],[0,8.5]], [90, 101], [90, 101]]

###############
## internals ##
###############

# define all paths absolute
abs_dir = os.path.dirname(__file__)

# load simulations
simout = pd.read_csv(os.path.join(abs_dir, f'simulations-variate_parameters-{parameter_name}.csv'), header=0, index_col=[0, 1, 2], parse_dates=True)
dates = simout.index.get_level_values('date').unique().values
parameter_values = simout.index.get_level_values(parameter_name).unique().values

########################
## visualise only epi ##
########################

fig, ax = plt.subplots(nrows=1, ncols=len(countries), figsize=(8.3, 11.7/3.3), sharex=True)

ylimit=[[0, 6.3],[0,7]]

for j, (country,IC_multiplier) in enumerate(zip(countries, IC_multipliers)):
    for val,color in zip(parameter_values,colors):
        ## Simulations
        ax[j].plot(range(len(dates)), simout.loc[(country, val, slice(None)), 'Ih']*IC_ratio*IC_multiplier,
                    color=color, linewidth=2)

    ## HCS capacity
    # lines
    ax[j].axhline(IC_beds_nominal[j]/population[j]*100000*IC_multiplier, xmin=0, xmax=1,
                        linestyle='--', color='black', linewidth=1)
    #ax[j].axhline(IC_beds_extended[j]/population[j]*100000*IC_multiplier, xmin=0, xmax=1,
    #                    linestyle='--', color='black', linewidth=1)
    # text
    ax[j].text(x=location_IC_annotation, y=(IC_beds_nominal[j])/population[j] *
                100000*IC_multiplier+0.20, s=f'nominal IC capacity', size=10)
    #ax[j].text(x=location_IC_annotation, y=(IC_beds_extended[j])/population[j] *
    #            100000*IC_multiplier+0.20, s=f'extended IC capacity', size=10)   
    ## hide spines
    ax[j].spines[['right', 'top']].set_visible(False)
    ## y-axis
    # ylimits
    ax[j].set_ylim(ylimit[j])
    # ylabels
    if j==0:
        ax[j].set_ylabel(ylabels[0], size=14)
    # no yticks for IC load
    ax[j].set_yticks([])
    # align y labels
    posx=-0.025
    ax[j].yaxis.set_label_coords(posx, 0.5)
    ## x-axis
    # xticksize
    ax[j].tick_params(axis='both', which='major', labelsize=14)
    # xlabels
    ax[j].set_xlabel('time (days)', size=14)
    # legend
    #if j == len(countries)-1:
    #    ax[j].legend(parameter_values, title=f'{parameter_name_fancy} ({parameter_unit})', framealpha=1, loc=4)
    # title
    ax[j].set_title(country_names[j], size=14)

plt.tight_layout()
plt.savefig(f'simulations-variate_parameters-{parameter_name}-epi_only.pdf')
#plt.show()
plt.close()

#########################
## visualise scenarios ##
#########################

# make figure
fig, ax = plt.subplots(nrows=len(states), ncols=len(countries), figsize=(11.7, 8.3), sharex=True)

for i,(state,ylimit,ylabel) in enumerate(zip(states,ylimits,ylabels)):
    for j, (country,IC_multiplier) in enumerate(zip(countries, IC_multipliers)):
        for val,color in zip(parameter_values,colors):
            # visualise scenarios
            if state == 'Ih':
                ax[i,j].plot(range(len(dates)), simout.loc[(country, val, slice(None)), state]*IC_ratio*IC_multiplier, linewidth=2, color=color)
            else:
                ax[i,j].plot(range(len(dates)), simout.loc[(country, val, slice(None)), state], color=color, linewidth=2)
        
        ## HCS capacity
        if i==0:
        # lines
            ax[0, j].axhline(IC_beds_nominal[j]/population[j]*100000*IC_multiplier, xmin=0, xmax=1,
                                linestyle='--', color='black', linewidth=1)
            #ax[0, j].axhline(IC_beds_extended[j]/population[j]*100000*IC_multiplier, xmin=0, xmax=1,
            #                    linestyle='--', color='black', linewidth=1)
            # text
            ax[0, j].text(x=location_IC_annotation, y=(IC_beds_nominal[j])/population[j] *
                        100000*IC_multiplier+0.20, s=f'nominal IC capacity', size=10)
            #ax[0, j].text(x=location_IC_annotation, y=(IC_beds_extended[j])/population[j] *
            #            100000*IC_multiplier+0.20, s=f'extended IC capacity', size=10)   
        ## hide spines
        ax[i,j].spines[['right', 'top']].set_visible(False)
        ## y-axis
        # ylimits
        if i == 0:
            ax[i,j].set_ylim(ylimit[j])
        else:
            ax[i,j].set_ylim(ylimit)
        # ylabels
        if j==0:
            ax[i,j].set_ylabel(ylabel, size=14)
        # no yticks for IC load
        if i==0:
            ax[i,j].set_yticks([])
        # align y labels
        posx=-0.075
        ax[0, j].yaxis.set_label_coords(posx, 0.5)
        ## x-axis
        # xlabels
        if i == len(states)-1:
            ax[i, j].set_xlabel('time (days)', size=14)
        # legend
        if ((i == len(states)-1) & (j == len(countries)-1)):
            ax[i, j].legend(parameter_values, title=f'{parameter_name_fancy} ({parameter_unit})', framealpha=1)
        # title
        if i==0:
            ax[i, j].set_title(country_names[j])

plt.tight_layout()
plt.savefig(f'simulations-variate_parameters-{parameter_name}.pdf')
plt.show()
plt.close()
