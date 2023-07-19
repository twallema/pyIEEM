import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyIEEM.data.utils import to_pd_interval
from pyIEEM.data.utils import smooth_contact_matrix, aggregate_contact_matrix, make_reciprocal

###############
## Load data ##
###############

data = pd.read_csv('comesf_raw_matrices.csv', index_col = [0,1,2,3,4,5,6], converters = {'age_x': to_pd_interval, 'age_y': to_pd_interval})
demography = pd.read_csv('../../../../../../data/interim/epi/demographic/age_structure_FR_2019.csv', index_col=[0,1]).squeeze().droplevel(0)
age_classes = list(data.index.get_level_values('age_x').unique().sort_values().values)
data = data.rename(columns={'desired_format': 'contacts'})
data = data.sort_index()

d = data.copy()

################################################################
## liquidate locations 'work_leisure_outdoor' and 'transport' ##
################################################################

## Transport contacts

# contacts < 20: week, False: school, week, True --> leisure_public, weekend, True/False: leisure_public
d.loc[('school', slice(None), slice(None), 'weekday', False, age_classes[0:4]), 'contacts'] += \
    data.loc[('transport', slice(None), slice(None), 'weekday', False, age_classes[0:4]), 'contacts'].values

d.loc[('leisure_public', slice(None), slice(None), 'weekday', True, age_classes[0:4]), 'contacts'] += \
    data.loc[('transport', slice(None), slice(None), 'weekday', True, age_classes[0:4]), 'contacts'].values
d.loc[('leisure_public', slice(None), slice(None), 'weekendday', slice(None), age_classes[0:4]), 'contacts'] += \
    data.loc[('transport', slice(None), slice(None), 'weekendday', slice(None), age_classes[0:4]), 'contacts'].values

# contacts 20 < age < 60: 
d.loc[('work_indoor', slice(None), slice(None), 'weekday', slice(None), age_classes[4:12]), 'contacts'] += \
    data.loc[('transport', slice(None), slice(None), 'weekday', slice(None), age_classes[4:12]), 'contacts'].values
d.loc[('leisure_public', slice(None), slice(None), 'weekendday', slice(None), age_classes[4:12]), 'contacts'] += \
    data.loc[('transport', slice(None), slice(None), 'weekendday', slice(None), age_classes[4:12]), 'contacts'].values

# contacts > 60: always to leisure
d.loc[('leisure_public', slice(None), slice(None), slice(None), slice(None), age_classes[12:]), 'contacts'] += \
    data.loc[('transport', slice(None), slice(None), slice(None), slice(None), age_classes[12:]), 'contacts'].values

## Work/Leisure outdoors

# contacts 20 < age < 60: week --> work, weekend --> leisure
d.loc[('work_indoor', slice(None), slice(None), 'weekday', slice(None), age_classes[4:12]), 'contacts'] += \
    data.loc[('work_leisure_outdoor', slice(None), slice(None), 'weekday', slice(None), age_classes[4:12]), 'contacts'].values
d.loc[('leisure_private', slice(None), slice(None), 'weekendday', slice(None), age_classes[4:12]), 'contacts'] += \
    data.loc[('work_leisure_outdoor', slice(None), slice(None), 'weekendday', slice(None), age_classes[4:12]), 'contacts'].values

# contacts < 20 or > 60: --> leisure_private
d.loc[('leisure_private', slice(None), slice(None), slice(None), slice(None), age_classes[0:4]), 'contacts'] += \
    data.loc[('work_leisure_outdoor', slice(None), slice(None), slice(None), slice(None), age_classes[0:4]), 'contacts'].values
d.loc[('leisure_private', slice(None), slice(None), slice(None), slice(None), age_classes[12:]), 'contacts'] += \
    data.loc[('work_leisure_outdoor', slice(None), slice(None), slice(None), slice(None), age_classes[12:]), 'contacts'].values

## remove work_leisure_outdoor, transport
d = d.reset_index()
d = d[((d['location'] != 'work_leisure_outdoor') & (d['location'] != 'transport'))]

## rename work_indoor --> work
d['location'] = np.where(d['location'] == 'work_indoor', 'work', d['location'])

## Remove the sector 'not_applicable'
d = d[d['sector'] != 'not_applicable']

## restore index
d = d.groupby(by=['location', 'sector', 'duration', 'type_day', 'vacation', 'age_x', 'age_y']).last()

##########################################################################
## Distribute the SPC contacts over week/weekend + vacation/no vacation ##
##########################################################################

# Compute ratio weekday/weekend contacts at work_indoor

# pre-made dataframe with desired formatting
names = ['sector', 'type_day', 'vacation']
sectors = d.reset_index()['sector'].unique()
type_days = d.reset_index()['type_day'].unique()
vacations = d.reset_index()['vacation'].unique()                  
iterables = [sectors, type_days, vacations]
ratios = pd.Series(index=pd.MultiIndex.from_product(iterables, names=names), name='ratio_contacts', dtype=float)                           

# We'll use the absolute number of contacts in the work_indoor category to obtain the ratios
abs_contacts = d.loc['work'].groupby(by=['sector', 'type_day', 'vacation', 'age_x', 'age_y']).sum()
for sector in sectors:
    ref = np.mean(np.sum(abs_contacts.loc[sector, 'weekday', False].values.reshape(2*[len(age_classes),]), axis=1))
    for type_day in type_days:
        for vacation in vacations:
            mat = abs_contacts.loc[sector, type_day, vacation].values.reshape(2*[len(age_classes),])
            ratios.loc[sector, type_day, vacation] = round(np.mean(np.sum(mat,axis=1))/ref,2)

# Redistribute 'SPC' contacts according to ratio weekday/weekendday vacation/no vacation on contacts 'work indoor'
for sector in sectors:
    for type_day in type_days:
        for vacation in vacations:
            r = ratios.loc[sector, type_day, vacation]
            if r == 0:
                r = ratios.groupby(by=['type_day', 'vacation']).mean().loc[type_day, vacation]
            d.loc[('SPC', sector, slice(None), type_day, vacation, slice(None), slice(None)), 'contacts'] = \
                    r*data.loc[('SPC', sector, slice(None), 'weekday', False, slice(None), slice(None)), 'contacts'].values

#########################
## Eliminate durations ##
#########################

## Write a function to integrate contact matrix, add to dataframe as a column
# absolute
d_absolute_contacts = d.groupby(by=['location', 'sector', 'type_day', 'vacation', 'age_x', 'age_y']).sum()
d_absolute_contacts = d_absolute_contacts.rename(columns={'contacts': 'absolute_contacts'})
d_absolute_contacts = d_absolute_contacts.fillna(0)
# integrated
duration_min = np.array([37.5, 10, 150, 2.5, 240], dtype=float)
d_integrated_contacts = d.groupby(by=['location', 'sector', 'type_day', 'vacation', 'age_x', 'age_y']).apply(lambda x: np.sum(duration_min*x.values.flatten())).to_frame(name = 'integrated_contacts')
d_integrated_contacts = d_integrated_contacts.fillna(0)
# merge
d = pd.concat([d_absolute_contacts, d_integrated_contacts], axis=1)

######################################
## Smooth 'work' and 'SPC' matrices ##
######################################

for location in ['SPC', 'work']:
    for sector in sectors:
        for type_day in type_days:
            for vacation in vacations:
                for colname in list(d.columns):
                    try:
                        mat = smooth_contact_matrix(d.loc[(location, sector, type_day, vacation), colname], 4, 3)['smoothed_contacts'].values
                    except:
                        # Verified seperation error arises because sum of matrix < 10e-09 --> set zeros
                        mat = np.zeros(len(age_classes)*len(age_classes))
                    d.loc[(location, sector, type_day, vacation), colname] = mat

## Add the SPC matrices to the work matrices (we'll keep the SPC)
d.loc[('work', slice(None), slice(None), slice(None), slice(None), slice(None)), 'absolute_contacts'] += \
    d.loc[('SPC', slice(None), slice(None), slice(None), slice(None), slice(None)), 'absolute_contacts'].values
d.loc[('work', slice(None), slice(None), slice(None), slice(None), slice(None)), 'integrated_contacts'] += \
    d.loc[('SPC', slice(None), slice(None), slice(None), slice(None), slice(None)), 'integrated_contacts'].values

## Fill all remaining nan
d = d.fillna(0)

###########################################################
## Compute week/weekend average and add it as a type_day ##
###########################################################

## Average weekday and weekendday and add it as a new level
avg = (5/7*d.loc[slice(None), slice(None), 'weekday', slice(None), slice(None), slice(None)] + \
        2/7*d.loc[slice(None), slice(None), 'weekendday', slice(None), slice(None), slice(None)]).reset_index()
avg['type_day'] = 'average'
d = pd.concat([avg, d.reset_index()]).groupby(by=['location', 'sector', 'type_day', 'vacation', 'age_x', 'age_y']).last()

###########################################################################
## Sector 'K' has small sample size ---> Average with highly similar 'M' ##
###########################################################################

# K is equal to the average of K and M --> small sample size for K, similar relative incidence in Belgium (Verbeeck) between sectors
d.loc[(slice(None), 'K', slice(None), slice(None), slice(None), slice(None)), :] = \
    d.loc[(slice(None), ['K','M'], slice(None), slice(None), slice(None), slice(None)), :].groupby(by=['location', 'type_day', 'vacation', 'age_x','age_y']).mean().values

##########################################################
## Reciprocity at home, leisure_private, leisure_public ##
##########################################################

locations = ['home', 'leisure_public', 'leisure_private']

print(f"Enforcing reciprocity on locations: '{locations}'\n")
for location in locations:
    print(f"\tlocation: '{location}'")
    for sector in d.index.get_level_values('sector').unique().values:
        for type_day in d.index.get_level_values('type_day').unique().values:
            for vacation in d.index.get_level_values('vacation').unique().values:
                for colname in d.columns:
                    mat = make_reciprocal(d.loc[location, sector, type_day, vacation, slice(None), slice(None)][colname], demography)
                    d.loc[(location, sector, type_day, vacation, slice(None), slice(None)), colname] = mat.values          
print('\ndone.\n')

##############################################
## Expand sectors to every level of NACE 21 ##
##############################################

## Expand result to every level of NACE 21

# Define expanded dataframe
locations = d.index.get_level_values('location').unique().values
sectors = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
type_days = d.index.get_level_values('type_day').unique().values
vacations = d.index.get_level_values('vacation').unique().values
age_x = d.index.get_level_values('age_x').unique().values
age_y = d.index.get_level_values('age_y').unique().values
iterables = [locations, sectors, type_days, vacations, age_x, age_y]
new_df = pd.DataFrame(index=pd.MultiIndex.from_product(iterables, names=d.index.names), columns=d.columns)

# Fill it in: sectors for which we have data
for sector in ['A', 'C', 'D', 'F', 'G', 'M', 'K']:
    new_df.loc[slice(None), sector, slice(None), slice(None), slice(None), slice(None)] = \
        d.loc[slice(None), sector, slice(None), slice(None), slice(None), slice(None)].values

# Fill it in: aggregated NACE 21 sectors
new_df.loc[slice(None), 'O', slice(None), slice(None), slice(None), slice(None)] = \
    d.loc[slice(None), 'O, N', slice(None), slice(None), slice(None), slice(None)].values
new_df.loc[slice(None), 'N', slice(None), slice(None), slice(None), slice(None)] = \
    d.loc[slice(None), 'O, N', slice(None), slice(None), slice(None), slice(None)].values
new_df.loc[slice(None), 'Q', slice(None), slice(None), slice(None), slice(None)] = \
    d.loc[slice(None), 'P, Q', slice(None), slice(None), slice(None), slice(None)].values
new_df.loc[slice(None), 'S', slice(None), slice(None), slice(None), slice(None)] = \
    d.loc[slice(None), 'S, T', slice(None), slice(None), slice(None), slice(None)].values
new_df.loc[slice(None), 'T', slice(None), slice(None), slice(None), slice(None)] = \
    d.loc[slice(None), 'S, T', slice(None), slice(None), slice(None), slice(None)].values

# Fill it in: assumptions
# B equal to C due to high similarity
new_df.loc[slice(None), 'B', slice(None), slice(None), slice(None), slice(None)] = \
    d.loc[slice(None), 'C', slice(None), slice(None), slice(None), slice(None)].values
# E is equal to F --> motivated by similar relative incidence in Belgium (Verbeeck)
new_df.loc[slice(None), 'E', slice(None), slice(None), slice(None), slice(None)] = \
    d.loc[slice(None), 'F', slice(None), slice(None), slice(None), slice(None)].values
# H is equal to C --> motivated by similar relative incidence in Belgium (Verbeeck) --> Sector H is very homogeneous, ranging from bus drivers or air hostesses, to solitary truckers, to warehouse employees.
new_df.loc[slice(None), 'H', slice(None), slice(None), slice(None), slice(None)] = \
    d.loc[slice(None), 'C', slice(None), slice(None), slice(None), slice(None)].values  
# I is equal to G --> highly similar customer-facing interactions
new_df.loc[slice(None), 'I', slice(None), slice(None), slice(None), slice(None)] = \
    d.loc[slice(None), 'G', slice(None), slice(None), slice(None), slice(None)].values  
# J is equal to M --> motivated by similar relative incidence in Belgium (Verbeeck)
new_df.loc[slice(None), 'J', slice(None), slice(None), slice(None), slice(None)] = \
    d.loc[slice(None), 'M', slice(None), slice(None), slice(None), slice(None)].values 
# L is equal to G --> highly similar customer-facing interactions
new_df.loc[slice(None), 'L', slice(None), slice(None), slice(None), slice(None)] = \
    d.loc[slice(None), 'G', slice(None), slice(None), slice(None), slice(None)].values
# R is equal to G --> highly similar customer-facing interactions
new_df.loc[slice(None), 'R', slice(None), slice(None), slice(None), slice(None)] = \
    d.loc[slice(None), 'G', slice(None), slice(None), slice(None), slice(None)].values

# Schools are an important exception: synthetic school work contacts are the contacts with people aged < 25 yo
new_df.loc[slice(None), 'P', slice(None), slice(None), slice(None), age_classes[:5]] = \
    d.loc[slice(None), 'P, Q', slice(None), slice(None), slice(None), age_classes[:5]].values
new_df = new_df.fillna(0)
d = new_df.copy()


########################
## Visualise matrices ##
########################

from matplotlib.colors import SymLogNorm

# settings
type_day = 'average'
vacation = False

# index levels
age_classes = d.index.get_level_values('age_x').unique().values
sectors = ['A', 'C', 'D', 'F', 'G', 'K', 'M', 'O', 'P', 'Q', 'S']
sectors = ['A', 'A', 'A', 'A'] + sectors
sectors_names = ['A', 'A', 'A', 'A'] + ['A', 'C', 'D', 'F', 'G', 'K', 'M', 'O, N', 'P', 'Q', 'S, T']
locations = ['home', 'leisure_private', 'leisure_public', 'school'] + len(sectors)*['work',]
locations_titles = ['Home', 'Leisure (private)', 'Leisure (public)', 'School'] + len(sectors)*['Workplace',]

for colname in list(d.columns):

    fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(8.3, 11.7))

    for i in range(len(axs.flatten())):
        ax = axs.flatten()[i]

        # heatmap
        m = d.loc[(locations[i], sectors[i], type_day, vacation, slice(None), slice(None)), colname].values.reshape([len(age_classes), len(age_classes)])
        if colname == 'absolute_contacts':
            vthresh = 0.25
            vmax = 10.1
        else:
            vthresh = 15
            vmax = 1010

        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        x = sns.heatmap(m, annot=False, fmt='.1f', ax=ax, square=True, cmap="mako",
                         annot_kws={"size":6}, vmin=0, vmax=vmax, norm=SymLogNorm(linthresh=vthresh, vmin=0, vmax=vmax),
                         cbar=True, cbar_ax=cbar_ax)
        
        # Ticks
        ax.xaxis.tick_top() # x axis on top
        ax.xaxis.set_label_position('top')

        if i < 3:
            ax.set_xticks(np.array(range(len(age_classes)))+0.5)
            ax.set_xticklabels(age_classes, rotation=75, size=6)
        else:
            ax.set_xticks(ticks=[])
            ax.set_xticklabels(labels=[])
        
        if i in [1,2,4,5,7,8,10,11,13,14]:
            ax.set_yticks(ticks=[])
            ax.set_yticklabels(labels=[])
        else:
            ax.set_yticks(np.array(range(len(age_classes)))+0.5)
            ax.set_yticklabels(age_classes, rotation=15, size=6)
        
        
        # Titles
        if i < 4:
            ax.set_title(f'{locations_titles[i]}', fontsize=11)
        else:
            ax.set_title(f'{locations_titles[i]} ({sectors_names[i]})', fontsize=11)

        # Number of contacts
        if colname == 'absolute_contacts':
            textstr = f'{np.mean(np.sum(m,axis=1)):.1f} contacts'
        else:
            textstr = f'{np.mean(np.sum(m,axis=1)):.0f} contact min.'
            
        props = dict(boxstyle='round', facecolor='black', alpha=1)
        ax.text(0.05, 0.12, textstr, transform=ax.transAxes, fontsize=8, color='white',
        verticalalignment='top', bbox=props)
        
    #fig.delaxes(axs[-1,-1])
    #fig.tight_layout(rect=[0, 0, .9, 1])

    plt.savefig(f'{colname}_{type_day}_{vacation}.png', dpi=600)
    plt.close()

#################
## Save result ##
#################

d.to_csv('comesf_formatted_matrices.csv')


