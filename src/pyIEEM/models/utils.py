import os
import numpy as np
import pandas as pd
from pyIEEM.data.utils import to_pd_interval, aggregate_contact_matrix
from pyIEEM.models.models import SIR

abs_dir = os.path.dirname(__file__)


def initialize_SIR(country, age_classes, spatial=True, contact_type='absolute_contacts'):

    # Get model parameters
    # ====================

    initial_states, parameters, coordinates = get_epi_params(country, age_classes, spatial, contact_type)

    # Construct TDPF
    # ==============

    # load NACE 21 composition per spatial patch
    if spatial == True:
        sectors = pd.read_csv(os.path.join(
            abs_dir, f'../../../data/interim/eco/labor_market_composition/sector_structure_by_work_{country}.csv'), index_col=[0, 1])['rel'].sort_index()
    else:
        sectors = pd.read_csv(os.path.join(
            abs_dir, f'../../../data/interim/eco/labor_market_composition/sector_structure_by_work_{country}.csv'), index_col=[0, 1])['abs']
        sectors = sectors.groupby(by='economic_activity').sum()/sectors.groupby(by='economic_activity').sum().sum()

    # load social contacts
    contacts = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/epi/contacts/matrices/FR/comesf_formatted_matrices.csv'), index_col=[0, 1, 2, 3, 4, 5],
                           converters={'age_x': to_pd_interval, 'age_y': to_pd_interval})

    # load national demography
    demography = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).groupby(by='age').sum().squeeze()

    # load fraction employees in workplace during pandemic
    f_workplace = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/epi/contacts/proximity/ermg_summary.csv'), index_col=[0])['workplace']

    # load and normalise leisure association vector (lav)
    lav = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/epi/contacts/proximity/ermg_summary.csv'), index_col=[0])['association_leisure']
    lav = lav/sum(lav)

    # load the number of employees in every sector of the NACE 64 from the national accounts
    f_employees = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/eco/national_accounts/{country}/other_accounts_{country}_NACE64.csv'), index_col=[0])['Number of employees (-)']

    # compute fraction of employees in NACE 64 sector as a percentage of its NACE 21 sector
    f_employees = f_employees.reset_index()
    f_employees['NACE 21'] = f_employees['index'].str[0]
    f_employees = f_employees.rename(columns={'index': 'NACE 64'})
    f_employees = f_employees.groupby(['NACE 21', 'NACE 64'])['Number of employees (-)'].sum().reset_index()
    f_employees['fraction_NACE21'] = f_employees['Number of employees (-)'] / f_employees.groupby('NACE 21')['Number of employees (-)'].transform('sum')
    f_employees = f_employees.drop(columns = ['NACE 21', 'Number of employees (-)']).set_index('NACE 64').squeeze()
    
    # NACE 64 to NACE 21 conversion matrix
    convmat = pd.read_csv(os.path.join(abs_dir, f'../../../data/interim/eco/misc/conversion_matrix_NACE64_NACE21.csv'), index_col=[0], header=[0])
    NACE64_coordinates = convmat.index.values
    convmat = convmat.fillna(0).values

    from pyIEEM.models.TDPF import make_social_contact_function
    social_contact_function = make_social_contact_function(age_classes, demography, contact_type, contacts, sectors, f_workplace, lav, False, f_employees, convmat).get_contacts

    # define economic policies
    economic_policy = pd.Series(0, index=NACE64_coordinates, dtype=float)

    # add TDPF parameters to dictionary
    parameters.update({'social_policy': 1, 'economic_policy': economic_policy})

    # Initialize model
    # ================

    model = SIR(initial_states, parameters, coordinates=coordinates, time_dependent_parameters={'N': social_contact_function})

    return model

def get_epi_params(country, age_classes, spatial, contact_type):
    """
    A function to load, format and return the epidemiological model's parameters, excluding time-dependent parameters

    input
    =====

    country: str
        'BE' or 'SWE'

    age_classes: pd.IntervalIndex
        desired number of age groups in model

    spatial: bool
        True: provincial-level simulation. False: national simulation

    contact_type: str
        'absolute_contacts' or 'integrated_contacts'

    output
    ======

    initial_states: dict
        Dictionary containing the non-zero initial values of the epidemiological model

    parameters: dict
        Dictionary containing the (non time-dependent) parameters of the epidemiological model

    coordinates: dict
        Dictionary containing the dimension names and corresponding coordinates of the epidemiological model
    """

    # demography
    # ==========

    # Load, format, sort alphabetically
    demography = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1])
    demography = demography.groupby(by=['spatial_unit', pd.cut(
        demography.index.get_level_values('age').values, age_classes)], sort=True).sum()
    demography.index.rename(['spatial_unit', 'age_class'], inplace=True)

    if spatial == True:
        # spatial units
        spatial_units = demography.index.get_level_values(
            'spatial_unit').unique().values
        # demography and inital condition
        S = demography.unstack('spatial_unit').values
    elif spatial == False:
        # spatial units
        spatial_units = [country, ]
        # demography and inital condition
        S = np.expand_dims(demography.groupby(
            by='age_class').sum().unstack('age_class').values, axis=1)

    # recurrent mobility
    # ==================

    # Load and sort alphabetically
    mob = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/epi/mobility/{country}/recurrent_mobility_normactive_{country}.csv'), index_col=0)
    mob = mob.loc[sorted(mob.index), sorted(mob.columns)]

    if spatial == True:
        mob = mob.values
    elif spatial == False:
        # compute size active population per province
        demography = pd.read_csv(os.path.join(
            abs_dir, f'../../../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1])
        active_population = demography.groupby(by=['spatial_unit', pd.cut(demography.index.get_level_values(
            'age').values, pd.IntervalIndex.from_tuples([(15, 65),], closed='left'))]).sum().droplevel(1)
        # compute fraction of active population with a job and multiply it with the size of the active population
        mob = np.ones([1, 1], dtype=float)*sum(mob.sum(axis=1).values *
                                               active_population.squeeze().values/active_population.sum().values[0])

    # default contacts
    # ================

    # Load and sort labor market structure alphabetically
    sectors = pd.read_csv(os.path.join(abs_dir, f'../../../data/interim/eco/labor_market_composition/sector_structure_by_work_{country}.csv'),
                          index_col=[0, 1])['rel'].sort_index()

    # Load social contacts
    contacts = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/epi/contacts/matrices/FR/comesf_formatted_matrices.csv'), index_col=[0, 1, 2, 3, 4, 5],
                           converters={'age_x': to_pd_interval, 'age_y': to_pd_interval})

    # other matrices (daytype='average', vacation=False, absolute contacts)
    # gather
    N_other = contacts.loc['home', 'A', 'average', False,
                           slice(None), slice(None)][contact_type]
    for location in ['leisure_public', 'leisure_private', 'school']:
        N_other += contacts.loc[location, 'A', 'average', False,
                                slice(None), slice(None)][contact_type].values
    # convert to right demography
    N_other = aggregate_contact_matrix(N_other, age_classes, pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).groupby(by=['age']).sum().squeeze())

    if spatial == True:
        # work contacts
        N_work_array = np.zeros(
            [len(age_classes), len(age_classes), len(spatial_units)])
        for i, prov in enumerate(spatial_units):
            N_work = sectors.loc[prov, 'A']*contacts.loc['work', 'A',
                                                         'average', False, slice(None), slice(None)][contact_type]
            for sector in [x for x in contacts.index.get_level_values('sector').unique().values if x != 'A']:
                N_work += sectors.loc[prov, sector]*contacts.loc['work', sector,'average', False, slice(None), slice(None)][contact_type].values
            # demographic conversion
            N_work = aggregate_contact_matrix(N_work, age_classes, pd.read_csv(os.path.join(
                abs_dir, f'../../../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).groupby(by=['age']).sum().squeeze())
            # to np.array
            N_work_array[:, :, i] = N_work.values.reshape(
                2*[len(age_classes),])
        N_work = N_work_array
        # other contacts to np.array
        N_other = np.tile(np.expand_dims(N_other.values.reshape(
            2*[len(age_classes),]), axis=2), len(spatial_units))

    elif spatial == False:
        # compute economic structure at national level
        sectors = pd.read_csv(os.path.join(
            abs_dir, f'../../../data/interim/eco/labor_market_composition/sector_structure_by_work_{country}.csv'), index_col=[0, 1])['abs']
        sectors = sectors.groupby(by='economic_activity').sum(
        )/sectors.groupby(by='economic_activity').sum().sum()
        # work contacts
        N_work = sectors.loc['A']*contacts.loc['work', 'A', 'average',
                                               False, slice(None), slice(None)][contact_type]
        for sector in [x for x in contacts.index.get_level_values('sector').unique().values if x != 'A']:
            N_work += sectors.loc[sector]*contacts.loc['work', sector, 'average',
                                                       False, slice(None), slice(None)][contact_type].values
        # demographic conversion
        N_work = aggregate_contact_matrix(N_work, age_classes, pd.read_csv(os.path.join(
            abs_dir, f'../../../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).groupby(by=['age']).sum().squeeze())
        # to np.array
        N_work = np.expand_dims(N_work.values.reshape(
            2*[len(age_classes),]), axis=2)
        # other contacts to np.array
        N_other = np.expand_dims(
            N_other.values.reshape(2*[len(age_classes),]), axis=2)

    # labor market composition
    # ========================

    # Load and sort alphabetically
    sectors = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/eco/labor_market_composition/sector_structure_by_work_{country}.csv'), index_col=[0, 1])['rel'].sort_index()

    if spatial == True:
        # convert to a np.array
        sectors = sectors.unstack('economic_activity').values
    elif spatial == False:
        # convert to national level
        sectors = pd.read_csv(os.path.join(
            abs_dir, f'../../../data/interim/eco/labor_market_composition/sector_structure_by_work_{country}.csv'), index_col=[0, 1])['abs']
        sectors = sectors.groupby(by='economic_activity').sum(
        )/sectors.groupby(by='economic_activity').sum().sum()
        # convert to a np.array
        sectors = np.expand_dims(sectors.values, axis=0)

    # disease parameters
    # ==================

    parameters = {'beta': 0.010,
                  'gamma': 5,
                  'N': {'other': N_other, 'work': N_work},
                  'G': mob,
                  }

    # initial condition and coordinates
    # =================================

    # default initial condition: one infected divided over every possible metapopulation
    initial_states = {'S': S,
                      'I': 1/len(age_classes)/len(mob)*np.ones(S.shape)}

    # Define coordinates
    coordinates = {'age_class': age_classes,
                   'spatial_unit': spatial_units}

    return initial_states, parameters, coordinates
