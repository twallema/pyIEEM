import os
from re import I
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# define colorscale
cmap = {"orange": "#E69F00", "light_blue": "#56B4E9",
        "green": "#009E73", "yellow": "#F0E442",
        "blue": "#0072B2", "red": "#D55E00",
        "pink": "#CC79A7", "black": "#000000"}

##############
## settings ##
##############

# start measures
start_measures = datetime(2020, 3, 21)
# compute start of measures in integer time
start_measures = (start_measures - datetime(2020, 2, 1))/timedelta(days=1)
# countries and states included
countries = ['BE', 'SWE']
# other settings
ICR = 0.162
population = [11.6e6, 10.4e6]
IC_beds_nominal = [1000, 600]
IC_beds_extended = [2000, 1000]
location_IC_annotation = 25
IC_multipliers = [IC_beds_nominal[1]/IC_beds_nominal[i]
                  for i in range(len(IC_beds_nominal))]
country_names = ['Belgium', 'Sweden']
colors = [cmap['blue'], cmap['green'],
          cmap['red'], cmap['black'], cmap['pink']]
ylabels = ['IC load (beds)', 'Labor\ncompensation (%)'] # 'Gross aggregated output (%)'
ylimits = [[[0, 6.3], [0, 7]], [55, 101], [55, 101]]

###############
## load data ##
###############

# define all paths absolute
abs_dir = os.path.dirname(__file__)
# load simulations
simout = pd.read_csv(os.path.join(abs_dir, f'simulations-prevention_paradox.csv'),
                     header=0, index_col=[0, 1, 2], parse_dates=True)
dates = simout.index.get_level_values('date').unique().values
length_measures_list = simout.index.get_level_values(
    'length_measures').unique().values
# load number of inhabitants
inhabitants = pd.read_csv(os.path.join(
    abs_dir, f'../data/interim/epi/demographic/age_structure_BE_2019.csv'), index_col=[0, 1]).sum().values[0]

##########################
## print tabulated data ##
##########################

print('\n')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('%% print mean economic damage and cumulative IC beds %%')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')

states = ['x', 'l', 'Hin']
data = np.zeros([len(countries), len(length_measures_list), len(states)])
for i, country in enumerate(countries):
    for j, length_measures in enumerate(length_measures_list):
        for k, state in enumerate(states):
            if state in ['x', 'l']:
                data[i, j, k] = 100 - \
                    simout.loc[(country, length_measures,
                                slice(None)), state].mean()
            else:
                data[i, j, k] = (inhabitants/100000)*ICR*simout.loc[(country,
                                                                     length_measures, slice(None)), state].sum()

print(data[:, :, 0])
print('\n')
print(data[:, :, 1])
print('\n')
print(data[:, :, 2])
print('\n')
print(data[:, :, 0]/data[:, :, 2])
print('\n')
print(data[:, :, 1]/data[:, :, 2])

#########################
## visualise scenarios ##
#########################

print('\n%%%%%%%%%%%%%%%%%%%%')
print('%% making figures %%')
print('%%%%%%%%%%%%%%%%%%%%\n')

print('IC load only\n')
states = ['Ih']

# make figure
fig, ax = plt.subplots(nrows=1, ncols=len(countries), figsize=(8.3, 11.7/3.3))

for i, (state, ylimit, ylabel) in enumerate(zip(states, ylimits, ylabels)):
    for j, (country, IC_multiplier) in enumerate(zip(countries, IC_multipliers)):
        for length_measures, color in zip(length_measures_list, colors):
            # visualise scenarios
            ax[j].plot(range(len(dates)), simout.loc[(country, length_measures, slice(None)), state]*ICR*IC_multiplier,
                    alpha=0.9, linewidth=2, color=color)

        # HCS capacity
        # lines
        ax[j].axhline(IC_beds_nominal[j]/population[j]*100000*IC_multiplier, xmin=0, xmax=1,
                    linestyle='--', color='black', linewidth=1)
        # text
        if j == 0:
            ax[j].text(x=location_IC_annotation, y=(IC_beds_nominal[j])/population[j] *
                    100000*IC_multiplier+0.15, s=f'nominal IC capacity', size=9)  # s=f'nominal IC capacity: {IC_beds_nominal[j]} beds'

        # Lockdown length
        # 2 months
        ax[j].text(x=96, y=ylimit[j][1] + 0.1, s=f'2 m.', size=7, rotation=30)
        # 3 months
        ax[j].text(x=96+1*28, y=ylimit[j][1] + 0.1,
                s=f'3 m.', size=7, rotation=30)
        # 4 months
        ax[j].text(x=96+2*28, y=ylimit[j][1] + 0.1,
                s=f'4 m.', size=7, rotation=30)
        # 5 months
        ax[j].text(x=96+3*28, y=ylimit[j][1] + 0.1,
                s=f'5 m.', size=7, rotation=30)

        # y-axis
        # ylimits
        ax[j].set_ylim(ylimit[j])
        # ylabels
        if j == 0:
            ax[j].set_ylabel(ylabel, size=14)
        # no yticks for IC load
        ax[j].set_yticks([])
        # align y labels
        posx = -0.105
        ax[j].yaxis.set_label_coords(posx, 0.5)
        # x-axis
        # xlabels
        ax[j].set_xlabel('time (days)', size=14)
        # eliminate spines
        ax[j].spines[['right', 'top']].set_visible(False)
        # title
        ax[j].set_title(country_names[j], size=14, y=1.1)
        # xlimit
        ax[j].set_xlim([20, None])
        # xticksize
        ax[j].tick_params(axis='both', which='major', labelsize=14)
        # legend
        # if j == 1:
        #    ax[j].legend((np.array(length_measures_list)/28).astype(int), title=f'Lockdown (m.)', framealpha=1, loc='lower left')
        # align y labels
        for j in range(2):
            posx = -0.025
            ax[j].yaxis.set_label_coords(posx, 0.5)

    # do shading
    for j, (country, IC_multiplier) in enumerate(zip(countries, IC_multipliers)):
        for length_measures, color in zip(length_measures_list, colors):
            ax[j].axvspan(start_measures, start_measures +
                        length_measures, color='black', alpha=0.05)

plt.tight_layout()
plt.savefig(f'simulations-prevention_paradox_hosponly.pdf')
plt.show()
plt.close()

print('IC load and economy\n')
states = ['Ih', 'l']

# make figure
fig, ax = plt.subplots(nrows=len(states), ncols=len(countries), figsize=(8.3, 11.7/2.5), sharex=True)

for i, (state, ylimit, ylabel) in enumerate(zip(states, ylimits, ylabels)):
    for j, (country, IC_multiplier) in enumerate(zip(countries, IC_multipliers)):
        for length_measures, color in zip(length_measures_list, colors):
            # convert to IC load for state Ih
            sim = simout.loc[(country, length_measures, slice(None)), state]
            if state == 'Ih':
                sim *= ICR*IC_multiplier
            # visualise mean
            ax[i, j].plot(range(len(dates)), sim, color=color, linewidth=2, alpha=0.9, )
        
        if i == 0:
            # HCS capacity
            # lines
            ax[i,j].axhline(IC_beds_nominal[j]/population[j]*100000*IC_multiplier, xmin=0, xmax=1,
                        linestyle='--', color='black', linewidth=1)
            # text
            if j == 0:
                ax[i,j].text(x=location_IC_annotation, y=(IC_beds_nominal[j])/population[j] *
                        100000*IC_multiplier+0.15, s=f'nominal IC capacity', size=9)  # s=f'nominal IC capacity: {IC_beds_nominal[j]} beds'

            # Lockdown length
            # 2 months
            ax[i,j].text(x=96, y=ylimit[j][1] + 0.1, s=f'2 m.', size=7, rotation=30)
            # 3 months
            ax[i,j].text(x=96+1*28, y=ylimit[j][1] + 0.1,
                    s=f'3 m.', size=7, rotation=30)
            # 4 months
            ax[i,j].text(x=96+2*28, y=ylimit[j][1] + 0.1,
                    s=f'4 m.', size=7, rotation=30)
            # 5 months
            ax[i,j].text(x=96+3*28, y=ylimit[j][1] + 0.1,
                    s=f'5 m.', size=7, rotation=30)

        # y-axis
        # ylimits
        if i == 0:
            ax[i,j].set_ylim(ylimit[j])
        else:
            ax[i,j].set_ylim(ylimit)
        # ylabels
        if j == 0:
            ax[i,j].set_ylabel(ylabel, size=14)
        # no yticks for IC load
        if i == 0:
            ax[i,j].set_yticks([])
        # align y labels
        #posx = -50
        #ax[i,j].yaxis.set_label_coords(posx, 0.5)
        # x-axis
        # xlabels
        ax[i,j].set_xlabel('time (days)', size=14)
        # eliminate spines
        ax[i,j].spines[['right', 'top']].set_visible(False)
        # title
        if i == 0:
            ax[i,j].set_title(country_names[j], size=14, y=1.1)
        # xlimit
        ax[i,j].set_xlim([20, None])
        # xticksize
        ax[i,j].tick_params(axis='both', which='major', labelsize=14)
        # align y labels
        for j in range(len(countries)):
            posx = -0.20
            ax[i,j].yaxis.set_label_coords(posx, 0.5)

    # do shading
    if i ==0:
        for j, (country, IC_multiplier) in enumerate(zip(countries, IC_multipliers)):
            for length_measures, color in zip(length_measures_list, colors):
                ax[i,j].axvspan(start_measures, start_measures +
                            length_measures, color='black', alpha=0.05)

plt.tight_layout()
plt.savefig(f'simulations-prevention_paradox_full.pdf')
plt.show()
plt.close()
