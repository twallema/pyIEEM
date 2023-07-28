import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pyIEEM.models.models import SIR
from pyIEEM.data.utils import to_pd_interval, aggregate_contact_matrix, convert_age_stratified_property

abs_dir = os.path.dirname(__file__)

#################################
## Initialisation of the model ##
#################################

def initialize_model(country, age_classes, spatial, simulation_start, contact_type='absolute_contacts', ):

    # Get model parameters
    # ====================

    initial_states, parameters, coordinates = get_epi_params(country, age_classes, spatial, contact_type)

    # Construct social contact TDPF
    # =============================

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

    # memory length
    l=6*28

    from pyIEEM.models.TDPF import make_social_contact_function
    social_contact_function = make_social_contact_function(age_classes, demography, contact_type, contacts, sectors, f_workplace, lav, False, f_employees, convmat, simulation_start, l).get_contacts

    # define economic policies
    economic_closures = pd.Series(1, index=NACE64_coordinates, dtype=float)

    # add TDPF parameters to dictionary
    parameters.update({'tau': 14, 'social_restrictions': 1, 'economic_closures': economic_closures})

    # Construct seasonality TDPF
    # ==========================

    from pyIEEM.models.TDPF import make_seasonality_function
    seasonality_function = make_seasonality_function()
    parameters.update({'amplitude': 0.18})

    # Initialize model
    # ================

    model = SIR(initial_states, parameters, coordinates=coordinates, time_dependent_parameters={'N': social_contact_function, 'beta': seasonality_function})

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

    # infectivity
    parameters = {'beta': 0.024}

    # durations
    parameters.update({'alpha': 4.5,
                      'gamma': 0.7,
                      'delta': 5,
                      'epsilon': 14,
                      'N': {'other': N_other, 'work': N_work},
                      'G': mob,
                  })

    # fractions
    s = pd.Series(index=pd.IntervalIndex.from_tuples([(0, 12), (12, 120)], closed='left'),
                    data=np.array([0.56, 1]), dtype=float) #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8260804/
    h = pd.Series(index=pd.IntervalIndex.from_tuples([(0, 12), (12, 18), (18, 25), (25, 35), (35, 45), (45, 55), (55, 65), (65, 75), (75, 85), (85,120)], closed='left'),
                    data=np.array([0.01, 0.01, 0.015, 0.025, 0.03, 0.06, 0.12, 0.45, 0.95, 0.99]), dtype=float)
    a = pd.Series(index=pd.IntervalIndex.from_tuples([(0, 20), (20, 40), (40, 60), (60, 80), (80,85), (85, 120)], closed='left'),
                         data=np.array([0.82, 0.78, 0.70, 0.65, 0.35, 0.01]), dtype=float)   
    m = pd.Series(index=pd.IntervalIndex.from_tuples([(0, 10), (10, 20), (20, 30), (30, 40), (40,50), (50, 60), (60,70), (70,80), (80,120)], closed='left'),
                         data=np.array([0.000, 0.012, 0.015, 0.027, 0.041, 0.080, 0.164, 0.266, 0.404]), dtype=float)
       
    # convert to right age groups
    demography = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).groupby(by='age').sum().squeeze()
    s = convert_age_stratified_property(s, age_classes, demography)  
    h = convert_age_stratified_property(h, age_classes, demography)
    a = convert_age_stratified_property(a, age_classes, demography)
    m = convert_age_stratified_property(m, age_classes, demography)

    parameters.update({
        's': np.expand_dims(s.values, axis=1),
        'h': np.expand_dims(h.values, axis=1),
        'a': np.expand_dims(a.values, axis=1),
        'm': np.expand_dims(m.values, axis=1),
    })

    # mobility and social contact
    parameters.update({
        'N': {'other': N_other, 'work': N_work},
        'G': mob,
    })

    # initial condition and coordinates
    # =================================

    # default initial condition: one infected divided over every possible metapopulation
    initial_states = {'S': S,
                      'E': 1/len(age_classes)/len(mob)*np.ones(S.shape)}

    # Define coordinates
    coordinates = {'age_class': age_classes,
                   'spatial_unit': spatial_units}

    return initial_states, parameters, coordinates

########################################
## Time-dependent Parameter Functions ##
########################################

def ramp_fun(t, t_start, l, N_old, N_new):
    """

    input
    =====

    t : timestamp
        current date

    t_start : timestamp
        start of ramp

    l : timestamp
        length of ramp

    N_old : float/np.array
        old policy
    
    N_new : float/np.array
        new policy

    output
    ======

    N_t : float/np.array
        interpolation between N_old and N_new
    """
    if t_start < t < t_start+timedelta(days=l):
        return N_old + ((N_new-N_old)/l)*((t-t_start)/timedelta(days=1))
    else:
        return N_new

from dateutil.easter import easter
def is_Belgian_school_holiday(d):
    """
    A function returning 'True' if a given date is a school holiday or primary and secundary schools in Belgium
    Tertiary education, which starts Sept. 12 and has exam periods in Dec-Jan and May-June are not considered.
    
    Input
    -----
    
    d: datetime.datetime
        Current simulation date
    
    Returns
    -------
    
    is_Belgian_school_holiday: bool
        True: date `d` is a school holiday for primary and secundary schools
    """
    
    # Pre-allocate a vector containing the year's holiday weeks
    holiday_weeks = []
    
    # Herfstvakantie
    holiday_weeks.append(44)
    
    # Extract date of easter
    d_easter = easter(d.year)
    # Convert from datetime.date to datetime.datetime
    d_easter = datetime(d_easter.year, d_easter.month, d_easter.day)
    # Get week of easter
    w_easter = d_easter.isocalendar().week

    # Default logic: Easter holiday starts first monday of April
    # Unless: Easter falls after 04-15: Easter holiday ends with Easter
    # Unless: Easter falls in March: Easter holiday starts with Easter
    if d_easter >= datetime(year=d.year,month=4,day=15):
        w_easter_holiday = w_easter - 1
    elif d_easter.month == 3:
        w_easter_holiday = w_easter + 1
    else:
        w_easter_holiday = datetime(d.year, 4, (8 - datetime(d.year, 4, 1).weekday()) % 7).isocalendar().week
    holiday_weeks.append(w_easter_holiday)
    holiday_weeks.append(w_easter_holiday+1)

    # Krokusvakantie
    holiday_weeks.append(w_easter-6)

    # Extract week of Christmas
    # If Christmas falls on Saturday or Sunday, Christams holiday starts week after
    w_christmas_current = datetime(year=d.year,month=12,day=25).isocalendar().week
    if datetime(year=d.year,month=12,day=25).isoweekday() in [6,7]:
        w_christmas_current += 1
    w_christmas_previous = datetime(year=d.year-1,month=12,day=25).isocalendar().week
    if datetime(year=d.year-1,month=12,day=25).isoweekday() in [6,7]:
        w_christmas_previous += 1
    # Christmas logic
    if w_christmas_previous == 52:
        if datetime(year=d.year-1, month=12, day=31).isocalendar().week != 53:
            holiday_weeks.append(1)   
    if w_christmas_current == 51:
        holiday_weeks.append(w_christmas_current)
        holiday_weeks.append(w_christmas_current+1)
    if w_christmas_current == 52:
        holiday_weeks.append(w_christmas_current)
        if datetime(year=d.year, month=12, day=31).isocalendar().week == 53:
            holiday_weeks.append(w_christmas_current+1)

    # Define Belgian Public holidays
    public_holidays = [
        datetime(year=d.year, month=1, day=1),       # New Year
        d_easter + timedelta(days=1),                # Easter monday
        datetime(year=d.year, month=5, day=1),       # Labor day
        d_easter + timedelta(days=40),               # Acension Day
        datetime(year=d.year, month=7, day=21),      # National holiday
        datetime(year=d.year, month=8, day=15),      # Assumption Mary
        datetime(year=d.year, month=11, day=1),      # All Saints
        datetime(year=d.year, month=11, day=11),     # Armistice
        datetime(year=d.year, month=12, day=25),     # Christmas
    ]
    
    # Logic
    if ((d.isocalendar().week in holiday_weeks) | \
            (d in public_holidays)) | \
                ((datetime(year=d.year, month=7, day=1) <= d < datetime(year=d.year, month=9, day=1))):
        return True
    else:
        return False

def aggregate_simplify_contacts(contact_df, age_classes, demography, contact_type):
    """
    A function to demographically convert the contacts to the right age classes and simplify the (very large) pd.DataFrame `contact_df`

    input
    =====

    contact_df: pd.DataFrame
        contacts obtained from Beraud (2015)

    age_classes: pd.IntervalIndex
        age classes of the epidemiological model

    demography: pd.Series
        demography of country under study.

    contact_type: str
        'absolute_contacts'/'absolute_contacts_less_5_min' versus 'integrated_contacts'/'integrated_contacts_less_5_min'

    output
    ======

    N_home: pd.Series
        Home contacts. Index: [day_type, vacation, age_x, age_y]. Values: `contact_type`

    N_leisure_private: pd.Series
        Private leisure contacts. Idem.

    N_leisure_public: pd.Series
        Public leisure contacts. Idem.

    N_school: pd.Series
        School contacts. Idem.

    N_work: pd.Series
        Work contacts. Index: [sector, day_type, vacation, age_x, age_y]. Values: `contact_type`
    """

    # Determine if a demographic conversion is necessary (this is computationally demanding and takes approx. 1 min.)
    if all(age_classes == pd.IntervalIndex.from_tuples([(0,5),(5,10),(10,15),(15,20),(20,25),(25,30),(30,35),(35,40),(40,45),(45,50),(50,55),(55,60),(60,65),(65,70),(70,75),(75,80),(80,120)], closed='left')):
        convert = False
    else:
        convert = True

    matrices=[]
    for location in contact_df.index.get_level_values('location').unique().values:
        if location not in ['SPC', 'work']:
            # output dataframe
            type_days = contact_df.index.get_level_values('type_day').unique().values
            vacations = contact_df.index.get_level_values('vacation').unique().values
            age_x = age_y = age_classes
            iterables = [type_days, vacations, age_x, age_y]
            contact = pd.Series(index=pd.MultiIndex.from_product(iterables, names=['type_day', 'vacation', 'age_x', 'age_y']), name=location, dtype=float)                           
            for type_day in type_days:
                for vacation in vacations:
                    m = contact_df.loc[location, 'A', type_day, vacation, slice(None), slice(None)][contact_type]
                    if convert:
                        m = aggregate_contact_matrix(m, age_classes, demography)
                    contact.loc[type_day, vacation, slice(None), slice(None)] = m.values
            matrices.append(contact)

        elif location == 'work':
            # output dataframe
            sectors = contact_df.index.get_level_values('sector').unique().values
            type_days = contact_df.index.get_level_values('type_day').unique().values
            vacations = contact_df.index.get_level_values('vacation').unique().values
            age_x = age_y = age_classes
            iterables = [sectors, type_days, vacations, age_x, age_y]
            contact = pd.Series(index=pd.MultiIndex.from_product(iterables, names=['sector', 'type_day', 'vacation', 'age_x', 'age_y']), name=location, dtype=float)                           
            for sector in sectors:
                for type_day in type_days:
                    for vacation in vacations:
                        m = contact_df.loc[location, sector, type_day, vacation, slice(None), slice(None)][contact_type]
                        if convert:
                            m = aggregate_contact_matrix(m, age_classes, demography)
                        contact.loc[sector, type_day, vacation, slice(None), slice(None)] = m.values
            matrices.append(contact)

    return (*matrices,)