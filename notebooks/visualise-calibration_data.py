import random
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from matplotlib.ticker import MaxNLocator
from pyIEEM.data.data import get_economic_data, get_hospitalisation_incidence

abs_dir = os.path.dirname(__file__)

# define colorscale
cmap = {"orange" : "#E69F00", "light_blue" : "#56B4E9",
        "green" : "#009E73", "yellow" : "#F0E442",
        "blue" : "#0072B2", "red" : "#D55E00",
        "pink" : "#CC79A7", "black" : "#000000"}

##########################
## change settings here ##
##########################

start_visualisation = datetime(2020, 2, 21)
end_visualisation = datetime(2021, 2, 19)

#########################
## load model and data ##
#########################

# load data BE and SWE
data_BE_epi = get_hospitalisation_incidence('BE')
data_SWE_epi = get_hospitalisation_incidence('SWE')
# load economic data BE and SWE
data_BE_eco_GDP = get_economic_data('GDP', 'BE', relative=True)
data_BE_eco_employment = get_economic_data('employment', 'BE', relative=True)
data_SWE_eco_GDP = get_economic_data('GDP', 'SWE', relative=True)
data_SWE_eco_employment = get_economic_data('employment', 'SWE', relative=True)
# load number of demographies
demographies=[]
for country in ['BE', 'SWE']:
    demographies.append(pd.read_csv(os.path.join(
                        abs_dir, f'../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).groupby(by='spatial_unit').sum().squeeze())

################################
## print quarterly aggregates ##
################################

# print(100-100*data_BE_eco_GDP.resample('Y').mean())
# print(100-100*data_BE_eco_employment.resample('D').interpolate(method='linear').resample('Y').mean())

# print(100-100*data_SWE_eco_GDP.resample('Y').mean())
# print(100-100*data_SWE_eco_employment.resample('Y').mean())

################################################
## visualise calibration (epi + eco national) ##
################################################

titles = ['Belgium', 'Sweden']
countries = ['BE', 'SWE']

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8.3, 11.7/2), sharex=True)
for i, (data_epi, data_eco_GDP, data_eco_employment, country, demography) in enumerate(zip([data_BE_epi, data_SWE_epi], [data_BE_eco_GDP, data_SWE_eco_GDP], [data_BE_eco_employment, data_SWE_eco_employment], ['BE', 'SWE'], demographies)):
    
    ## epidemiological
    data = data_epi.loc[slice(start_visualisation, end_visualisation)].groupby(by='date').sum()/demography.sum()*100000
    
    # print hosp incidence when lockdown was triggered
    if country == 'SWE':
        start_measures = datetime(2020,3,11)
        data = data.resample('D').interpolate(method='linear')
        print(data.loc[datetime(2020,3,11)])
    else:
        start_measures = datetime(2020,3,15)
        print(data.loc[datetime(2020,3,15)])

    if country == 'BE':
        data = data.ewm(span=3).mean()

    ax[0, i].plot(data.loc[slice(None,start_measures)].index, data.loc[slice(None,start_measures)], linewidth=2, alpha=1, color='black')
    ax[0, i].plot(data.loc[slice(start_measures, None)].index, data.loc[slice(start_measures, None)], linewidth=2, alpha=1, color='black')

    # axes properties
    ax[0, i].set_xlim([start_visualisation, end_visualisation])
    ax[0, i].set_ylim([0, 7.5])
    ax[0, i].yaxis.set_major_locator(MaxNLocator(6))

    ### economic
    ## GDP
    data = data_eco_GDP.loc[slice(start_visualisation, end_visualisation)]
    ax[1, i].plot(data.index, 100*data, linewidth=2, marker='o', markersize=5, alpha=1, color='black', label='Gross output (%)')
    ## employment
    data = data_eco_employment.loc[slice(start_visualisation, end_visualisation)]
    ax[1, i].plot(data.index, 100*data, linewidth=2, marker='s', markersize=5, alpha=1, linestyle='--', color='black', label='Labor compensation (%)')
    # axes properties
    ax[1, i].set_xlim([start_visualisation, end_visualisation])
    ax[1, i].set_ylim([60, 102])
    ax[1, 1].legend(fontsize=11)

    ## Lockdown arrows BE
    t4 = pd.Timestamp(2020, 3, 15) 
    t5 = pd.Timestamp(2020, 5, 4)
    t8 = pd.Timestamp(2020, 7, 1)
    t9 = pd.Timestamp(2020, 8, 1)
    t10 = pd.Timestamp(2020, 9, 1)
    t11 = pd.Timestamp(2020, 10, 19)
    ylim=7.5
    label_lockdown_size = 10
    label_lockdown_move_up = 0.2
    arrow_height = ylim
    # First lockdown
    ax[0,0].annotate(text='', xy=(t4, arrow_height), xytext=(t5,arrow_height), \
                arrowprops=dict(arrowstyle='<-',linewidth=2))
    ax[0,0].text(t4 + pd.Timedelta(days=1), \
            ylim + label_lockdown_move_up, 'First national\nLockdown', size=label_lockdown_size)
    # Lockdown release
    ax[0,0].annotate(text='', xy=(t5, arrow_height), xytext=(t8,arrow_height), \
                arrowprops=dict(arrowstyle='<-',linewidth=2))
    # Local lockdown Antwerpen
    ax[0,0].annotate(text='', xy=(t9, arrow_height), xytext=(t10,arrow_height), \
                arrowprops=dict(arrowstyle='<-',linewidth=2))
    ax[0,0].text(t9 - pd.Timedelta(days=10), \
            ylim + label_lockdown_move_up, 'Local\nlockdown', size=label_lockdown_size)
    # Second lockdown
    label_lockdown1_move_right = -2.5 # days
    ax[0,0].annotate(text='', xy=(t11, arrow_height), xytext=(end_visualisation,arrow_height), \
                arrowprops=dict(arrowstyle='<-',linewidth=2))
    ax[0,0].text(t11 + pd.Timedelta(days=1), \
            ylim + label_lockdown_move_up, 'Second\nnational lockdown', size=label_lockdown_size)
    arrow_height = ylim

    ## Lockdown arrows SWE
    t1 = pd.Timestamp(2021, 1, 7) 
    # First restrictions
    ax[0,1].annotate(text='', xy=(t1, arrow_height), xytext=(end_visualisation,arrow_height), \
                arrowprops=dict(arrowstyle='<-',linewidth=2))
    ax[0,1].text(t1, ylim + label_lockdown_move_up, 'First\nRestrictions', size=label_lockdown_size)

    ## country text box
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    ax[0,i].text(0.05, 0.925, titles[i], transform=ax[0,i].transAxes, fontsize=11, color='black',
                verticalalignment='top', bbox=props)
            
    ## bounding box off
    ax[0,i].spines['right'].set_visible(False)
    ax[0,i].spines['top'].set_visible(False)
    ax[1,i].spines['right'].set_visible(False)
    ax[1,i].spines['top'].set_visible(False)
    
    ## x-axis
    # labels
    ax[0,i].set_xticks([datetime(2020,4,1), datetime(2020,7,1), datetime(2020,10,1), datetime(2021,1,1)])

    ## y-axis
    # y ticks
    # no yticks for IC load
    ax[0,1].set_yticks([])
    ax[1,1].set_yticks([])
    # rotate labels
    for tick in ax[1,i].get_xticklabels():
        tick.set_rotation(20)

    # ylabels left hand side only
    if i == 0:
        ax[0, i].set_ylabel('Hospital incidence\nper 100K inhab. (-)', fontsize=11)
        ax[1, i].set_ylabel('Gross output (%)\nLabor compensation (%)', fontsize=11)

plt.tight_layout()
plt.savefig(
    f'data_calibration.pdf')
plt.show()
plt.close()