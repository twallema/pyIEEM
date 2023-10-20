import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# define colorscale
cmap = {"orange": "#E69F00", "light_blue": "#56B4E9",
        "green": "#009E73", "yellow": "#F0E442",
        "blue": "#0072B2", "red": "#D55E00",
        "pink": "#CC79A7", "black": "#000000"}

##############
## settings ##
##############

end_visualisation = datetime(2020, 9, 15)
location_IC_annotation = datetime(2020, 6, 25)
scenarios = ['L1', 'L2b', 'L3b', 'L4a', 'L4b']
custom_scenario_labels = ['P1', 'P2', 'P3', 'P4a', 'P4b']
colors = [cmap['blue'], cmap['red'],
          cmap['green'], cmap['black'], cmap['black'], cmap['orange']]
linestyles = ['-', '-', '-', '-', '--']
# 'Gross\n aggregated output (%)',
ylabels = ['IC load per\n100K inhab. (-)', 'Labor income (%)']
ylims = [[0, 29], [60, 101], [60, 101]]
ICR = 0.162

###############
## internals ##
###############

# define all paths absolute
abs_dir = os.path.dirname(__file__)

# load simulations
simout = pd.read_csv(os.path.join(abs_dir, 'simulations_hypothetical_scenarios_BE.csv'), header=[
                     0, 1], index_col=[0, 1, 2], parse_dates=True)

# load number of inhabitants
inhabitants = pd.read_csv(os.path.join(
    abs_dir, f'../data/interim/epi/demographic/age_structure_BE_2019.csv'), index_col=[0, 1]).sum().values[0]

# derive variables
t_start_lockdowns = simout.index.get_level_values(
    't_start_lockdown').unique().values
t_start_lockdowns.sort()
t_start_lockdowns = t_start_lockdowns[3:]
dates = simout.index.get_level_values('date').unique().values
states = simout.columns.get_level_values('state').unique().values

# define titles (automatically)
# titles = []
# for t_start_lockdown in t_start_lockdowns:
#    dt = np.datetime_as_string(t_start_lockdown, unit='D')
#    titles.append(f'Measures imposed on\n{dt}')
# define titles (manually)
titles = ['Measures enforced earlier\non March 6, 2020',
          'Measures enforced earlier\non March 9, 2020', 'Measures enforced earlier\non March 12, 2020',
          'Measures enforced on\nMarch 15, 2020', 'Measures enforced later\non March 18, 2020']

titles = ['Measures enforced earlier\non March 12, 2020',
          'Measures enforced on\nMarch 15, 2020', 'Measures enforced later\non March 18, 2020']

print('\n')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('%% print quarterly aggregates %%')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')

# compute and store 2020 Q2 aggregates
states = ['x', 'l', 'Hin']
data = np.zeros([len(t_start_lockdowns), len(scenarios), len(states)])

for i, t_start_lockdown in enumerate(t_start_lockdowns):
    for j, scenario in enumerate(scenarios):
        for k, state in enumerate(states):
            if state in ['x', 'l']:
                data[i, j, k] = 100-simout.loc[(scenario, t_start_lockdown, slice(None)), (state, 'mean')].groupby(
                    by='date').last().resample('Q').mean().loc[datetime(2020, 6, 30)]
            else:
                data[i, j, k] = (inhabitants/100000)*ICR*simout.loc[(scenario, t_start_lockdown, slice(
                    None)), (state, 'mean')].groupby(by='date').last().resample('Q').sum().loc[datetime(2020, 6, 30)]

# print results
for k, state in enumerate(states):
    print(f'state: {state}')
    print(data[:, :, k])
    print('\n')

# print economic damage per occupied IC bed
print(data[:, :, 0]/data[:, :, -1])
print('\n')
print(data[:, :, 1]/data[:, :, -1])
print('\n')

# visualise marginale meeropbrengst
fig, ax = plt.subplots(figsize=(8.3, 11.7/4))
for i, t_start_lockdown in enumerate(t_start_lockdowns):
    ax.plot(scenarios, data[i, :, 0]/data[i, :, -1],
            label=np.datetime_as_string(t_start_lockdown, unit='D'), color=colors[i])
ax.spines[['right', 'top']].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('Implemented policy', size=12)
ax.set_ylabel('Ratio gross. agg. output\nover cumulative IC patients', size=12)

ax.legend(fontsize=12, title='Start lockdown', framealpha=1, title_fontsize=12)
plt.tight_layout()
plt.show()
plt.close()

######################
## make full figure ##
######################

print('%%%%%%%%%%%%%%%%%%%%')
print('%% making figures %%')
print('%%%%%%%%%%%%%%%%%%%%\n')

linewidth = 2
states = ['Ih', 'l']

fig, ax = plt.subplots(nrows=len(states), ncols=len(
    t_start_lockdowns), figsize=(8.3, 11.7/2.5), sharex=True) # small figure: figsize=(8.3, 11.7/2.5)

for i, state in enumerate(states):
    for j, t_start_lockdown in enumerate(t_start_lockdowns):
        for k, (scenario, custom_scenario_label) in enumerate(zip(scenarios, custom_scenario_labels)):
            # convert to IC load for state Ih
            sim = simout.loc[(scenario, t_start_lockdown,
                              slice(None)), (state, slice(None))]
            if state == 'Ih':
                sim *= ICR
            # visualise mean
            ax[i, j].plot(dates, sim[state, 'mean'], color=colors[k], linestyle=linestyles[k],
                          linewidth=linewidth, alpha=0.9, label=custom_scenario_label)
            # shade uncertainty
            # ax[i, j].fill_between(dates, sim[state, 'lower'], sim[state, 'upper'], color=colors[k], alpha=0.2)

        # HCS capacity
        if i == 0:
            # lines
            ax[i, j].axhline(1000/11.4e6*100000, xmin=0, xmax=1,
                             linestyle='--', color='black', linewidth=1)
            ax[i, j].axhline(2000/11.4e6*100000, xmin=0, xmax=1,
                             linestyle='--', color='black', linewidth=1)
            # text
            ax[i, -1].text(x=location_IC_annotation, y=1060/11.4e6 *
                           100000, s='nominal\nIC capacity', size=8)
            ax[i, -1].text(x=location_IC_annotation, y=2060/11.4e6 *
                           100000, s='extended\nIC capacity', size=8)

        # title
        if i == 0:
            ax[i, j].set_title(titles[j], size=12)
        # ticksize
        # ax[i,j].tick_params(axis='both', which='major', labelsize=14)
        # Remove spines
        ax[i, j].spines[['right', 'top']].set_visible(False)
        # y-axis
        # labels
        if j == 0:
            ax[i, j].set_ylabel(ylabels[i], size=12)
        # ylimits
        ax[i, j].set_ylim(ylims[i])
        # yticks
        if j != 0:
            ax[i, j].set_yticks([])
        # x-axis
        # xlabels
        if i == len(states)-1:
            for tick in ax[i, j].get_xticklabels():
                tick.set_rotation(30)
        # maximum number of xticks
        ax[i, j].set_xticks(
            [datetime(2020, 4, 1), datetime(2020, 6, 1), datetime(2020, 8, 1)])

        # ax[i, j].xaxis.set_major_locator(MaxNLocator(6))
        # legend only on last plot
        if ((i == len(states)-1) & (j == len(t_start_lockdowns)-1)):
            ax[i, j].legend(framealpha=1)
        # x-range
        if i == len(states)-1:
            ax[i, j].set_xlim([dates[0], end_visualisation])

plt.tight_layout()
plt.savefig('hypothetical_scenarios_BE.pdf')
plt.show()
plt.close()

#########################
## make reduced figure ##
#########################
