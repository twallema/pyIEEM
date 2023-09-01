import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# define colorscale
cmap = {"orange" : "#E69F00", "light_blue" : "#56B4E9",
        "green" : "#009E73", "yellow" : "#F0E442",
        "blue" : "#0072B2", "red" : "#D55E00",
        "pink" : "#CC79A7", "black" : "#000000"}

##############
## settings ##
##############

# start measures
start_measures = datetime(2020, 3, 21)
# compute start of measures in integer time
start_measures = (start_measures - datetime(2020, 2, 1))/timedelta(days=1)
# countries and states included
countries = ['BE', 'SWE']
states = ['Ih',]
# other settings
ICR = 0.162
population = [11.6e6, 10.4e6]
IC_beds_nominal = [1000, 600]
IC_beds_extended = [2000, 1000]
location_IC_annotation = 175
IC_multipliers = [IC_beds_nominal[1]/IC_beds_nominal[i] for i in range(len(IC_beds_nominal))]
country_names = ['Belgium', 'Sweden']
colors = [cmap['blue'], cmap['green'], cmap['red'], cmap['black']]
ylabels = ['IC load (beds)', 'Gross aggregated output (%)', 'Labor compensation (%)']
ylimits = [[0, 7], [80, 101], [80, 101]]

###############
## internals ##
###############

# define all paths absolute
abs_dir = os.path.dirname(__file__)
# load simulations
simout = pd.read_csv(os.path.join(abs_dir, f'simulations-prevention_paradox.csv'), header=0, index_col=[0, 1, 2], parse_dates=True)
dates = simout.index.get_level_values('date').unique().values
length_measures_list = simout.index.get_level_values('length_measures').unique().values

#########################
## visualise scenarios ##
#########################

# make figure
fig, ax = plt.subplots(nrows=1, ncols=len(countries), figsize=(11.7, 8.3/2))

for i,(state,ylimit,ylabel) in enumerate(zip(states,ylimits,ylabels)):
    for j, (country, IC_multiplier) in enumerate(zip(countries, IC_multipliers)):
        for length_measures, color in zip(length_measures_list, colors):
            # visualise scenarios
            ax[j].plot(range(len(dates)), simout.loc[(country, length_measures, slice(None)), state]*ICR*IC_multiplier, color=color)
            
        # HCS capacity
        # lines
        ax[j].axhline(IC_beds_nominal[j]/population[j]*100000*IC_multiplier, xmin=0, xmax=1,
                            linestyle='--', color='black', linewidth=1)
        #ax[j].axhline(IC_beds_extended[j]/population[j]*100000*IC_multiplier, xmin=0, xmax=1,
        #                    linestyle='--', color='black', linewidth=1)
        # text
        ax[j].text(x=location_IC_annotation, y=(IC_beds_nominal[j])/population[j] *
                    100000*IC_multiplier+0.20, s=f'nominal IC capacity: {IC_beds_nominal[j]} beds', size=8)
        #ax[j].text(x=location_IC_annotation, y=(IC_beds_extended[j])/population[j] *
        #            100000*IC_multiplier+0.20, s=f'extended IC capacity: {IC_beds_extended[j]} beds', size=8) 
        ## y-axis
        # ylimits
        ax[j].set_ylim(ylimit)
        # ylabels
        if j==0:
            ax[j].set_ylabel(ylabel)
        # no yticks for IC load
        ax[j].set_yticks([])
        # align y labels
        posx=-0.105
        ax[j].yaxis.set_label_coords(posx, 0.5)
        ## x-axis
        # xlabels
        ax[j].set_xlabel('time (days)')
        # title
        ax[j].set_title(country_names[j])
        # legend
        if j == 1:
            ax[j].legend((np.array(length_measures_list)/28).astype(int), title=f'Lockdown (m.)', framealpha=1, loc='lower left')
    
    # do shading
    for j, (country, IC_multiplier) in enumerate(zip(countries, IC_multipliers)):
        for length_measures, color in zip(length_measures_list, colors):        
            ax[j].axvspan(start_measures, start_measures+length_measures, color='black', alpha=0.05)


plt.savefig(f'simulations-prevention_paradox.png', dpi=300)
plt.show()
plt.close()