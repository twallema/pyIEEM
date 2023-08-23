import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# define colorscale
cmap = {"orange" : "#E69F00", "light_blue" : "#56B4E9",
        "green" : "#009E73", "yellow" : "#F0E442",
        "blue" : "#0072B2", "red" : "#D55E00",
        "pink" : "#CC79A7", "black" : "#000000"}

##############
## settings ##
##############

end_visualisation = datetime(2021, 2, 1)
location_IC_annotation = datetime(2020, 7, 1)
scenarios = ['L1', 'L2b', 'L3b', 'L4a', 'L4b']
colors = [cmap['blue'], cmap['red'], cmap['green'], cmap['black'], cmap['black']]
linestyles = ['-', '-', '-', '-', '--']
ylabels = ['Hospital incidence (-)', 'ICU load (-)', 'Productivity (%)', 'Employment (%)']
ylims = [[0, 13.5], [0, 25], [60, 102], [60, 102]]

###############
## internals ##
###############

# define all paths absolute
abs_dir = os.path.dirname(__file__)

# load simulations
simout = pd.read_csv(os.path.join(abs_dir, 'simulations_hypothetical_scenarios_BE.csv'), header=[0, 1], index_col=[0, 1, 2], parse_dates=True)

# derive variables
t_start_lockdowns = simout.index.get_level_values('t_start_lockdown').unique().values
dates = simout.index.get_level_values('date').unique().values
states = simout.columns.get_level_values('state').unique().values

# define titles
titles = []
for t_start_lockdown in t_start_lockdowns:
    dt = np.datetime_as_string(t_start_lockdown, unit='D')
    titles.append(f'Measures imposed on\n{dt}')

# compute quarterly aggregates
for i, t_start_lockdown in enumerate(t_start_lockdowns):
    for j, scenario in enumerate(scenarios):
        for state in ['x', 'l']:
            print(f"\nmeasures imposed on {dt}")
            print(f"quarterly reduction of '{state}' in scenario '{scenario}'")
            print(100-simout.loc[(scenario, t_start_lockdown, slice(None)), (state, 'mean')].groupby(by='date').last().resample('Q').mean())

# make figure
fig, ax = plt.subplots(nrows=len(states), ncols=len(t_start_lockdowns), figsize=(11.7, 8.3), sharex=True)

for i, state in enumerate(states):
    for j, t_start_lockdown in enumerate(t_start_lockdowns):
        for k, scenario in enumerate(scenarios):
            # convert to IC load for state Ih
            sim = simout.loc[(scenario, t_start_lockdown,slice(None)), (state, slice(None))]
            if state == 'Ih':
                sim *= 0.168
            # visualise mean
            ax[i, j].plot(dates, sim[state, 'mean'], color=colors[k], linestyle=linestyles[k],
                          linewidth=1.5, alpha=0.9, label=scenario)
            # shade uncertainty
            #ax[i, j].fill_between(dates, sim[state, 'lower'], sim[state, 'upper'], color=colors[k], alpha=0.2)

        # HCS capacity
        if i == 1:
            # lines
            ax[i, j].axhline(1000/11.4e6*100000, xmin=0, xmax=1,
                             linestyle='--', color='black', linewidth=1)
            ax[i, j].axhline(2000/11.4e6*100000, xmin=0, xmax=1,
                             linestyle='--', color='black', linewidth=1)
            # text
            ax[i, 0].text(x=location_IC_annotation, y=1060/11.4e6 *
                          100000, s='nominal IC capacity', size=8)
            ax[i, 0].text(x=location_IC_annotation, y=2060/11.4e6 *
                          100000, s='extended IC capacity', size=8)

        # title indicating lockdown startdate
        if i == 0:
            ax[i, j].set_title(titles[j])

        # y-axis
        # labels
        if j == 0:
            ax[i, j].set_ylabel(ylabels[i])
        # ylimits
        ax[i, j].set_ylim(ylims[i])

        # x-axis
        # xlabels
        if i == len(states)-1:
            for tick in ax[i, j].get_xticklabels():
                tick.set_rotation(30)
        # maximum number of xticks
        ax[i, j].xaxis.set_major_locator(MaxNLocator(7))
        # legend only on last plot
        if ((i == len(states)-1) & (j == len(t_start_lockdowns)-1)):
            ax[i,j].legend(framealpha=1)
        # x-range
        if i == len(states)-1:
            ax[i,j].set_xlim([dates[0], end_visualisation])

plt.tight_layout()
plt.show()
plt.close()
