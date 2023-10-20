import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pyIEEM.models.TDPF import gompertz

abs_dir = os.path.dirname(__file__)

# define colorscale
cmap = {"orange" : "#E69F00", "light_blue" : "#56B4E9",
        "green" : "#009E73", "yellow" : "#F0E442",
        "blue" : "#0072B2", "red" : "#D55E00",
        "pink" : "#CC79A7", "black" : "#000000"}

# settings go here
country = 'BE'
start_visualisation = datetime(2020, 2, 1)
end_visualisation = datetime(2020, 8, 1)

########################
## load hospital load ##
########################

data = pd.read_csv(os.path.join(
                    abs_dir, f'../data/interim/epi/cases/hospital_incidence_{country}.csv'), index_col=[0], parse_dates=True)
# simplify spelling
data.loc[data['PROVINCE'] == 'WestVlaanderen', 'PROVINCE'] = 'West-Vlaanderen'
data.loc[data['PROVINCE'] == 'OostVlaanderen', 'PROVINCE'] = 'Oost-Vlaanderen'
data.loc[data['PROVINCE'] == 'BrabantWallon', 'PROVINCE'] = 'Brabant Wallon'
data.loc[data['PROVINCE'] == 'VlaamsBrabant', 'PROVINCE'] = 'Vlaams-Brabant'
data.loc[data['PROVINCE'] == 'Liège', 'PROVINCE'] = 'Liege'
# cut of at start of 2021
data = data.loc[slice(None, datetime(2022, 1, 1)), :]
# make an empty dataframe with all date-province combinations as index
names = ['date', 'spatial_unit']
dates = data.reset_index()['DATE'].unique()
provinces = data['PROVINCE'].unique()
iterables = [dates, provinces]
desired_data = pd.Series(index=pd.MultiIndex.from_product(iterables, names=names), name='Ih', dtype=float).sort_index()     
# slice right column and set the index
data = data[['PROVINCE', 'TOTAL_IN']].groupby(by=['DATE', 'PROVINCE']).sum()
data = data.squeeze().rename('Ih')
data.index.names = ['date', 'spatial_unit']
# merge dataframes
data = desired_data.combine_first(data).fillna(0)
# retain only national hospital load
data = data.groupby(by='date').sum().loc[slice(start_visualisation, end_visualisation)]
# load demography per spatial patch
demography = pd.read_csv(os.path.join( abs_dir, f'../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).sum()
# normalize to incidence per 100K
data /= demography.values
data *= 100000
# convert to dataframe
data = data.to_frame()

#########################
## append M-parameters ##
#########################

halflife_data = 8
halflife_EMA = 30

data['M'] = gompertz(data.ewm(halflife=halflife_EMA).mean(), 10, 0.18)

######################
## visualise result ##
######################

# EMA
fig,ax = plt.subplots(figsize=(8,2.5))
ax.plot(range(-len(data.index),0), data.Ih.ewm(halflife=halflife_data).mean(), color='black', linewidth=3, label=r'Actual hosp. load')
ax.plot(range(-len(data.index),0), data.Ih.ewm(halflife=halflife_EMA).mean(), color=cmap['red'], linewidth=3, label=r'EMA hosp. load')
ax.legend(framealpha=1, fontsize=13)
ax.set_ylabel('Norm. hosp. load\n$(Q_{hosp})^g(t)$', size=13)
ax.set_xlabel('Days in the past', size=13)
ax.set_xticks([0, -50, -100, -150])
ax.set_yticks([0, 15, 30, 45])
ax.set_ylim([0,50])
ax.spines[['right', 'top']].set_visible(False)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.savefig('EMA.pdf')
#plt.show()
plt.close()

# mentality function
fig,ax = plt.subplots(figsize=(8,2.5))
ax.plot(range(-len(data.index),0), data.M, color=cmap['red'], linewidth=3, label='$M_{leisure}(t)$')
ax.set_ylabel('Behavioral change\n'+r'$\widebar{M}_x^g(t)$', size=13)
ax.set_xlabel('Days in the past', size=13)
ax.set_xticks([0, -50, -100, -150])
ax.spines[['right', 'top']].set_visible(False)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.savefig('M_time.pdf')
#plt.show()
plt.close()

# hysteresis
fig,ax = plt.subplots(figsize=(8,2.8))
ax.plot(data.Ih.ewm(halflife=halflife_data).mean(), data.M, color='black', linewidth=3, label=r'Phase trajectory')
ax.plot(data.Ih.ewm(halflife=halflife_EMA).mean(), data.M, color=cmap['red'], linewidth=3, label=r'Gompertz model')
ax.legend(framealpha=1, fontsize=13)
ax.set_xlabel('Norm. hosp. load\n$(Q_{hosp})^g(t)$', size=13)
ax.set_ylabel('Behavioral change\n'+r'$\widebar{M}_x^g(t)$', size=13)
ax.spines[['right', 'top']].set_visible(False)
# arrow downward
arrow_x = 24
arrow_y = 0.625
ax.annotate('', xy=(arrow_x, arrow_y), xytext=(arrow_x-1.3, arrow_y-0.06),
                arrowprops=dict(arrowstyle='-|>, head_width=0.5, head_length=1.5', lw=2, fc='black'), annotation_clip=False)
# arrow upward
arrow_x = 5
arrow_y = 0.49
ax.annotate('', xy=(arrow_x, arrow_y), xytext=(arrow_x+1.85, arrow_y+0.13),
                arrowprops=dict(arrowstyle='-|>, head_width=0.5, head_length=1.5', lw=2, fc='black'), annotation_clip=False)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.savefig('M_phasediagram.pdf')
plt.show()
plt.close()