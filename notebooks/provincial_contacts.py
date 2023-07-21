import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

abs_dir = os.path.dirname(__file__)

# settings
vacation = False
type_day = 'average'
type_contact = 'absolute_contacts'
country_suffixes = ['BE', 'SWE']

# Load contact matrices
contacts = pd.read_csv(os.path.join(abs_dir, '../data/interim/epi/contacts/matrices/FR/comesf_formatted_matrices.csv'), index_col=[0,1,2,3,4,5])
contacts = contacts.loc[slice(None), slice(None), type_day, vacation, slice(None), slice(None)][type_contact]

# Compute number of contacts outside of work (doesn't depend on economic activity)
age_classes = contacts.index.get_level_values('age_x').unique().values
baseline_n = 0
for location in ['home', 'leisure_public', 'leisure_private', 'school']:
    baseline_n += np.mean(np.sum(contacts.loc[location, 'A', slice(None), slice(None)].values.reshape(2*[len(age_classes),]), axis=1))

###############################
## Destination-base contacts ##
###############################

gpdf_modified = []
for i, country_suffix in enumerate(country_suffixes):
    ## Load data
    # Load geojson file and set index to names of spatial units
    gpdf = gpd.read_file(os.path.join(abs_dir, f'../data/interim/epi/shape/{country_suffix}.json'))
    gpdf.set_index('spatial_unit', inplace=True)
    gpdf.index.name = 'spatial_unit'
    # Load recurrent mobility matrices (normalized by total and not active population!)
    mob = pd.read_csv(os.path.join(abs_dir, f'../data/interim/epi/mobility/{country_suffix}/recurrent_mobility_normtotal_{country_suffix}.csv'), index_col=0)
    # Load total population
    pop = pd.read_csv(os.path.join(abs_dir, f'../data/interim/epi/demographic/age_structure_{country_suffix}_2019.csv'), index_col=[0,1]).groupby(by='spatial_unit').sum().reindex(mob.index)
    # Compute mobility-modified population
    workpop = np.matmul(np.transpose(mob), pop).squeeze()
    # Load NACE 21 composition at provincial level
    sectors = pd.read_csv(os.path.join(abs_dir, f'../data/interim/eco/labor_market_composition/sector_structure_by_work_{country_suffix}.csv'), index_col=[0,1])['rel']
    print(pop)
    ## desination-based average number of contacts 
    # Compute the number of contacts at work
    work_cont = pop.copy()
    for prov in sectors.index.get_level_values('spatial_unit').unique():
        n = 0
        for sector in sectors.index.get_level_values('economic_activity').unique():
            c = np.mean(np.sum(contacts.loc['work', sector, slice(None), slice(None)].values.reshape(2*[len(age_classes),]), axis=1))
            n += workpop.loc[prov]*sectors.loc[prov, sector]*c
        work_cont.loc[prov] = n 

    # Add baseline contacts and divide by the number of inhabitants to arive at the desination-based average number of contacts
    cont = pop.copy()
    cont.loc[slice(None)] = np.expand_dims(np.squeeze(work_cont.values)/np.squeeze(pop.values) + baseline_n, axis=1)
    # Assign to geopandas
    gpdf['n_contacts_destination'] = 0
    for prov in gpdf.index:
        gpdf.loc[prov, 'n_contacts_destination'] = cont.loc[prov].values
    
    ## Origin-based
    
    # Normalize every column of the mobility matrix so that it adds up to one (Then you know the distribution of where people came from to work in a given spatial patch)
    mob = mob.div(mob.sum(axis=0), axis=1)

    # Use mobility matrix to distribute the total number of contacts in a given spatial patch across the places where the people making those contacts came from
    c = np.zeros(len(gpdf.index))
    for prov in gpdf.index:
        c += np.squeeze(mob[prov].values)*work_cont.loc[prov,:].values
    cont = pop.copy()
    cont[slice(None)] = c

    # Compute per inhabitant
    cont.loc[slice(None)] = np.expand_dims(np.squeeze(cont.values)/np.squeeze(pop.values) + baseline_n, axis=1)

    # Assign to geopandas
    gpdf['n_contacts_origin'] = 0
    for prov in gpdf.index:
        gpdf.loc[prov, 'n_contacts_origin'] = cont.loc[prov].squeeze()

    # Save result
    gpdf_modified.append(gpdf)

row_titles = ['Origin-based number of contacts', 'Destination-based number of contacts']
row_columns = ['n_contacts_origin', 'n_contacts_destination']
props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=1)
vmin=16
vmax=24
# Plot data
fig,axes = plt.subplots(nrows=2, ncols=2, figsize=(6,12))
for i in range(2):
    for j in range(2):
        ax = axes[i,j]
        gpdf_modified[j].plot(column=row_columns[i], legend= ((i==1)&(j==1)), ax=ax, cmap='Blues', vmin=vmin, vmax=vmax)
        gpdf_modified[j].boundary.plot(ax=ax, color='black', linewidth=0.1, alpha=1)
        ax.axis('off')
# place a text box in upper left in axes coords
axes[0,0].text(0, 1.55, 'Origin-based number of contacts', transform=axes[0,0].transAxes, fontsize=11,
        verticalalignment='top', bbox=props)
axes[1,0].text(0, 1.55, 'Destination-based number of contacts', transform=axes[1,0].transAxes, fontsize=11,
        verticalalignment='top', bbox=props)
plt.savefig('n_contacts.png', dpi=600)
plt.show()
plt.close()