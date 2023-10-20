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

weigh_demographic = False
countries = ['BE', 'SWE']
country_names = ['Belgium', 'Sweden']
IC_ratio = 0.162
population = [11.6e6, 10.4e6]
IC_beds_nominal = [1000, 600]
IC_beds_extended = [2000, 1000]
IC_multipliers = [IC_beds_nominal[1]/IC_beds_nominal[i] for i in range(len(IC_beds_nominal))]
location_IC_annotation = 0
ylabels = ['IC load (beds)', 'Labor\ncompensation (%)']
ylims = [[0, 13.5], [87, 101], [87, 101]]
highlights = [['Brussels',], ['Stockholm']]

###############
## internals ##
###############

# define all paths absolute
abs_dir = os.path.dirname(__file__)

# load simulations
simouts=[]
dates=[]
spatial_units=[]
for country in countries:
    simout = pd.read_csv(os.path.join(abs_dir, f'simulations-hypothetical_spatial_spread_{country}-demographic_weighing_{weigh_demographic}.csv'), header=0, index_col=[0, 1], parse_dates=True)
    simouts.append(simout)
    dates.append(simout.index.get_level_values('date').unique().values)
    spatial_units.append(simout.index.get_level_values('spatial_unit').unique().values)

##########################
## visualise popdensity ##
##########################

# https://www.sciencedirect.com/science/article/pii/S0025556413001235?via%3Dihub
pop_density_BE = [254.50514263, 146.83923914, 3088.06815663, 142.204992, 117.19390958, 142.34518126, 26.63181018,
                    55.56176989, 200.43268078, 207.28383628, 148.23435747]
pop_density_SWE = [54.4, 10.3, 15.9, 61.5, 2.7, 34.8, 22, 23.9, 2.6, 35.8, 44.1, 125.7, 49, 364.9, 46.9, 16.1, 5, 11.4, 53.9, 72.5]           
# Gotland: originally position 4 (starting at one) with popdens 19

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8.3, 11.7/2.5), sharex=True)

for i, (simout, spatial_unit,country,popdens) in enumerate(zip(simouts, spatial_units,countries,[pop_density_BE, pop_density_SWE])):
    
    # load full demography
    demography = pd.read_csv(os.path.join(
        abs_dir, f'../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).groupby(by='spatial_unit').sum().squeeze()
    
    # append maximum peak IC load
    min_employment=[]
    max_ICU = []
    for su in spatial_unit:
        print(su, max(simout.loc[(su, slice(None)), 'Ih']*IC_ratio))
        if not su == 'Gotland':
            max_ICU.append(max(simout.loc[(su, slice(None)), 'Ih']*IC_ratio))
            min_employment.append(min(simout.loc[(su, slice(None)), 'l']))

    # visualise ICU data
    ax[0, i].scatter(popdens, 100*np.array(max_ICU)/(IC_beds_nominal[i]/sum(demography)*100000), color='black')
    ax[0, i].axhline(100, xmin=0, xmax=1, linestyle='--', color='black', linewidth=1)

    # visualise employment data
    #ax[1, i].scatter(demography.values/sum(demography), min_employment, color='black')
    ax[1, i].scatter(popdens, min_employment, color='black')

    # axes labels
    ax[1, i].set_xlabel(f"Population density (inhab./km2)", size=14)
    ax[0, 0].set_ylabel('Fraction of nominal\nIC capacity (%)', size=14)
    ax[1, 0].set_ylabel('Labor\ncompensation (%)', size=14)

    # axes limits
    ax[1, 0].set_xlim([0,300])
    ax[1, 1].set_xlim([0,300])
    ax[0, 0].set_ylim([20,160])
    ax[0, 1].set_ylim([20,160])
    ax[1, 0].set_ylim([80,101])
    ax[1, 1].set_ylim([80,101])

plt.tight_layout()
plt.savefig(
    'visualise-hypothetical_spatial_spread_1.pdf',
)
#plt.show()
plt.close()

#########################
## visualise scenarios ##
#########################

# make figure
fig, ax = plt.subplots(nrows=2, ncols=len(countries), figsize=(8.3, 11.7/2.5), sharex=True)

for j, (simout, date, spatial_unit, highlight, country,IC_multiplier) in enumerate(zip(simouts,dates,spatial_units,highlights,countries,IC_multipliers)):

    ## plot all data
    for su in spatial_unit:
        # correct plot order for highlights
        if su in highlight:
            color = '0.10'
            zorder = 1
        else:
            color = '0.7'
            zorder = -1
        if su != 'Gotland':
            # IC load
            ax[0,j].plot(range(len(date)), simout.loc[(su, slice(None)), 'Ih']*IC_ratio*IC_multiplier, color=color, linewidth=1.5, zorder=zorder)
            # GDP
            #ax[1,j].plot(range(len(date)), simout.loc[(su, slice(None)), 'x'], color=color, linewidth=1.5, zorder=zorder)
            # employment
            ax[1,j].plot(range(len(date)), simout.loc[(su, slice(None)), 'l'], color=color, linewidth=1.5, zorder=zorder)
            # append maximum peak IC load
            max_ICU.append(max(simout.loc[(su, slice(None)), 'Ih']*IC_ratio))

    ## HCS capacity
    # lines
    ax[0, j].axhline(IC_beds_nominal[j]/population[j]*100000*IC_multiplier, xmin=0, xmax=1,
                        linestyle='--', color='black', linewidth=1)
    ax[0, j].axhline(IC_beds_extended[j]/population[j]*100000*IC_multiplier, xmin=0, xmax=1,
                        linestyle='--', color='black', linewidth=1)
   # text
    ax[0, j].text(x=location_IC_annotation, y=(IC_beds_nominal[j])/population[j] *
                100000*IC_multiplier+0.20, s=f'nominal IC capacity', size=10)

    ax[0, j].text(x=location_IC_annotation, y=(IC_beds_extended[j])/population[j] *
                100000*IC_multiplier+0.20, s=f'extended IC capacity', size=10)                     

    # title
    ax[0,j].set_title(country_names[j], size=14)

    ## x-axis
    # maximum number of xticks
    ax[1, j].xaxis.set_major_locator(MaxNLocator(7))
    # rotate xticks
    #for tick in ax[2, j].get_xticklabels():
    #    tick.set_rotation(30)
    # xlabels
    ax[1, j].set_xlabel('time (days)', size=14)

    ## y-axis
    # labels and limits
    for i in range(2):
        if j == 0:
            ax[i, j].set_ylabel(ylabels[i], size=14)
        ax[i, j].set_ylim(ylims[i])  
    # no ticks with IC load
    ax[0,j].set_yticks([])
    # no yticks on right hand side of figure
    ax[1,1].set_yticks([])
    #ax[2,1].set_yticks([])
    # align y labels
    for j in range(2):
        posx=-0.075
        ax[0, j].yaxis.set_label_coords(posx, 0.5)
    # hide spines
    ax[0, 1].spines[['right', 'top']].set_visible(False)
    ax[1, 1].spines[['right', 'top']].set_visible(False)
    #ax[2, 1].spines[['right', 'top']].set_visible(False)
    ax[0, 0].spines[['right', 'top']].set_visible(False)
    ax[1, 0].spines[['right', 'top']].set_visible(False)
    #ax[2, 0].spines[['right', 'top']].set_visible(False)

plt.tight_layout()
plt.savefig(
    'visualise-hypothetical_spatial_spread_2.pdf'
)
#plt.show()
plt.close()

