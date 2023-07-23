import os
import numpy as np
import pandas as pd
from pyIEEM.data.utils import to_pd_interval, aggregate_contact_matrix
from pyIEEM.models.models import SIR

abs_dir = os.path.dirname(__file__)

def initialize_SIR(country, age_classes, spatial=True):

    ##############################################
    ## Demography, mobility, economic structure ##
    ##############################################

    # Load and format demographics (alphabetic)
    demography = pd.read_csv(os.path.join(abs_dir, f'../../../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0,1])
    demography = demography.groupby(by=['spatial_unit', pd.cut(demography.index.get_level_values('age').values, age_classes)], sort=True).sum()
    demography.index.rename(['spatial_unit', 'age_class'], inplace=True)
    
    # Load recurrent mobility matrices and sort alphabetically
    mob = pd.read_csv(os.path.join(abs_dir, f'../../../data/interim/epi/mobility/{country}/recurrent_mobility_normactive_{country}.csv'), index_col=0)
    mob = mob.loc[sorted(mob.index), sorted(mob.columns)]

    # Load NACE 21 composition at provincial level and sort alphabetically
    sectors = pd.read_csv(os.path.join(abs_dir, f'../../../data/interim/eco/labor_market_composition/sector_structure_by_work_{country}.csv'), index_col=[0,1])['rel'].sort_index()

    # Load social contacts
    contacts = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/epi/contacts/matrices/FR/comesf_formatted_matrices.csv'), index_col=[0,1,2,3,4,5],
                            converters={'age_x': to_pd_interval, 'age_y': to_pd_interval})

    # other matrices (daytype='average', vacation=False, absolute contacts)
    # gather
    N_other = contacts.loc['home', 'A', 'average', False, slice(None), slice(None)]['absolute_contacts']
    for location in ['leisure_public', 'leisure_private', 'school']:
        N_other += contacts.loc[location, 'A', 'average', False, slice(None), slice(None)]['absolute_contacts'].values
    # convert to right demography
    N_other = aggregate_contact_matrix(N_other, age_classes, pd.read_csv(os.path.join(abs_dir, f'../../../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0,1]).groupby(by=['age']).sum().squeeze())

    if spatial == True:

        # spatial units
        spatial_units = demography.index.get_level_values('spatial_unit').unique().values

        # demography and inital condition
        S = demography.unstack('spatial_unit').values
        I = np.ones(S.shape)

        # mobility
        mob = mob.values

        # work contacts
        N_work_array = np.zeros([len(age_classes), len(age_classes), len(spatial_units)])
        for i,prov in enumerate(spatial_units):
            N_work = sectors.loc[prov, 'A']*contacts.loc['work', 'A', 'average', False, slice(None), slice(None)]['absolute_contacts']
            for sector in [x for x in contacts.index.get_level_values('sector').unique().values if x != 'A']:
                N_work += sectors.loc[prov, sector]*contacts.loc['work', sector, 'average', False, slice(None), slice(None)]['absolute_contacts'].values
            # demographic conversion
            N_work = aggregate_contact_matrix(N_work, age_classes, pd.read_csv(os.path.join(abs_dir, f'../../../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0,1]).groupby(by=['age']).sum().squeeze())
            # to np.array
            N_work_array[:, :, i] = N_work.values.reshape(2*[len(age_classes),])
        N_work = N_work_array

        # economic structure
        sectors = sectors.unstack('economic_activity').values

        # other contacts
        N_other = np.tile(np.expand_dims(N_other.values.reshape(2*[len(age_classes),]), axis=2), len(spatial_units))

    elif spatial == False:

        # spatial units
        spatial_units = [country, ]

        # demography and inital condition
        S = np.expand_dims(demography.groupby(by='age_class').sum().unstack('age_class').values, axis=1)

        # compute size active population per province
        demography = pd.read_csv(os.path.join(abs_dir, f'../../../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0,1])
        active_population = demography.groupby(by=['spatial_unit', pd.cut(demography.index.get_level_values('age').values, pd.IntervalIndex.from_tuples([(15,65),], closed='left'))]).sum().droplevel(1)

        # compute fraction of active population with a job and multiply it with the size of the active population
        mob = np.ones([1,1], dtype=float)*sum(mob.sum(axis=1).values*active_population.squeeze().values/active_population.sum().values[0])

        # compute economic structure at national level
        sectors = pd.read_csv(os.path.join(abs_dir, f'../../../data/interim/eco/labor_market_composition/sector_structure_by_work_{country}.csv'), index_col=[0,1])['abs']
        sectors = sectors.groupby(by='economic_activity').sum()/sectors.groupby(by='economic_activity').sum().sum()

        # work contacts
        N_work = sectors.loc['A']*contacts.loc['work', 'A', 'average', False, slice(None), slice(None)]['absolute_contacts']
        for sector in [x for x in contacts.index.get_level_values('sector').unique().values if x != 'A']:
            N_work += sectors.loc[sector]*contacts.loc['work', sector, 'average', False, slice(None), slice(None)]['absolute_contacts'].values
        # demographic conversion
        N_work = aggregate_contact_matrix(N_work, age_classes, pd.read_csv(os.path.join(abs_dir, f'../../../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0,1]).groupby(by=['age']).sum().squeeze())
        # to np.array
        N_work = np.expand_dims(N_work.values.reshape(2*[len(age_classes),]), axis=2)

        # convert sectors to a np.array
        sectors = np.expand_dims(sectors.values, axis=0)

        # other contacts
        N_other = np.expand_dims(N_other.values.reshape(2*[len(age_classes),]), axis=2)

    ##########
    ## TDPF ##
    ##########

    # reload NACE 21 composition per patch
    if spatial == True:
        sectors = pd.read_csv(os.path.join(abs_dir, f'../../../data/interim/eco/labor_market_composition/sector_structure_by_work_{country}.csv'), index_col=[0,1])['rel'].sort_index()
    else:
        sectors = pd.read_csv(os.path.join(abs_dir, f'../../../data/interim/eco/labor_market_composition/sector_structure_by_work_{country}.csv'), index_col=[0,1])['abs']
        sectors = sectors.groupby(by='economic_activity').sum()/sectors.groupby(by='economic_activity').sum().sum()

    
    from pyIEEM.models.TDPF import make_social_contact_function
    social_contact_function = make_social_contact_function(contacts, sectors).get_contacts

    ######################
    ## Initialize model ##
    ######################

    # default initial condition: one infected divided over every possible metapopulation
    initial_states = {'S': S,
                      'I': 1/len(age_classes)/len(mob)*np.ones(S.shape)}

    # Define coordinates
    coordinates = {'age_class': age_classes,
                   'spatial_unit': spatial_units}

    # Define parameters
    parameters = {'beta': 0.007,
                  'gamma': 5,
                  'N': {'other': N_other, 'work': N_work},
                  'G': mob,
                  }

    model = SIR(initial_states, parameters, coordinates=coordinates)              

    return model