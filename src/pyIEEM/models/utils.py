import os
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pyIEEM.models.models import epidemic_model, epinomic_model
from pyIEEM.data.utils import to_pd_interval, aggregate_contact_matrix, convert_age_stratified_property

abs_dir = os.path.dirname(__file__)

###############################
## Initialise epinomic model ##
###############################

def initialize_epinomic_model(country, age_classes, spatial, simulation_start, contact_type='absolute_contacts',
                                prodfunc='half_critical', scenarios=False):

    # get default model parameters
    # ============================

    # get
    initial_states, parameters, coordinates = get_epi_params(country, age_classes, spatial, contact_type)
    st, par, coord = get_eco_params(country, prodfunc)
    # attach
    initial_states.update(st)
    parameters.update(par)
    coordinates.update(coord)
    
    # get calibrated epidemiological model states
    # ===========================================

    # no simulation start --> default: one exposed divided over all spatial patches and age groups
    if scenarios == 'hypothetical_spatial_spread':
        sim = xr.open_dataset(os.path.join(abs_dir, f'../../../data/interim/epi/initial_condition/{country}_INITIAL_CONDITION.nc'))
        for data_var in sim.keys():
            if spatial == True:
                initial_states.update({data_var: sim.sel(date=simulation_start)[data_var].values})   
            else:
                initial_states.update({data_var: np.expand_dims(sim.sum(dim='spatial_unit').sel(date=simulation_start)[data_var].values, axis=1)})   

    # add the IC multiplier
    # =====================

    # reference:
    n_SWE = 600
    n_BE = 1000
    IC_multiplier = n_SWE/n_BE

    # construct social contact TDPF (epidemic)
    # ========================================

    # get all necessary parameters
    parameters, demography, contacts, lmc_stratspace, lmc_strateco, f_workplace, f_remote, hesitancy, lav, f_employees, convmat = get_social_contact_function_parameters(parameters, country, spatial, scenarios)
    # define all relevant parameters of the social contact function TDPF here
    parameters.update({'l': 5, 'mu': 1, 'nu': 24, 'xi_work': 5, 'xi_eff': 0.50, 'xi_leisure': 5,
                        'pi_work': 0.02, 'pi_eff': 0.06, 'pi_leisure': 0.30})
    # make social contact function
    from pyIEEM.models.TDPF import make_social_contact_function
    social_contact_function = make_social_contact_function(IC_multiplier, age_classes, demography, contact_type, contacts, lmc_stratspace, lmc_strateco, f_workplace, f_remote, hesitancy, lav,
                                                            False, True, f_employees, convmat, simulation_start, country)
    
    # select right function
    if scenarios == False:
        if country == 'BE':
            social_contact_function = social_contact_function.get_contacts_BE
        else:
            social_contact_function = social_contact_function.get_contacts_SWE
    elif scenarios == 'hypothetical_policy':
        if country == 'BE':
            social_contact_function = social_contact_function.get_contacts_BE_scenarios
    elif scenarios == 'hypothetical_spatial_spread':
        # disable vacations
        social_contact_function = make_social_contact_function(IC_multiplier, age_classes, demography, contact_type, contacts, lmc_stratspace, lmc_strateco, f_workplace, f_remote, hesitancy, lav,
                                                                False, False, f_employees, convmat, simulation_start, country)
        social_contact_function = social_contact_function.get_contacts_trigger

    # construct seasonality TDPF (epidemic)
    # =====================================

    from pyIEEM.models.TDPF import make_seasonality_function
    seasonality_function = make_seasonality_function(country)

    parameters.update({'amplitude_BE': 0.20, 'peak_shift_BE': -14, 'amplitude_SWE': 0.20, 'peak_shift_SWE': 14}) 

    # construct labor supply shock TDPF (economic)
    # ============================================

    from pyIEEM.models.TDPF import make_labor_supply_shock_function
    ## lmc_strateco
    # get labor market composition (abs)
    lmc_stratspace = pd.read_csv(os.path.join(abs_dir, f'../../../data/interim/eco/labor_market_composition/sector_structure_by_work_{country}.csv'), index_col=[0, 1], header=0)['abs'].sort_index()
    # convert to the fraction of laborers in spatial patch 'i' working in sector 'X' of the total number of laborers working in sector 'X' (NACE 21)
    lmc_stratspace = lmc_stratspace/lmc_stratspace.groupby('economic_activity').transform('sum')
    # convert from NACE 21 to NACE 64 using the ratios found in the national accounts
    iterables = [lmc_stratspace.index.get_level_values('spatial_unit').unique().values, f_employees.index]
    names = ['spatial_unit', 'economic_activity']
    out = pd.Series(index=pd.MultiIndex.from_product(iterables, names=names), name='f_employees', dtype=float)
    for act in f_employees.index.values:
        out.loc[slice(None), act] = lmc_stratspace.loc[slice(None), act[0]].values # *f_employees.loc[act]
    lmc_strateco = out
    # convert if spatial is false
    if not spatial:
        lmc_strateco = lmc_strateco.groupby(by='economic_activity').sum()
    # load fraction employees in workplace during pandemic
    f_workplace = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/epi/contacts/proximity/ermg_summary.csv'), index_col=[0])['workplace']
    # load telework fraction observed during pandemic
    f_remote = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/epi/contacts/proximity/ermg_summary.csv'), index_col=[0])['remote']
    # load the number of employees in every sector of the NACE 64 from the national accounts
    f_employees = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/eco/national_accounts/{country}/other_accounts_{country}_NACE64.csv'), index_col=[0])['Number of employees (-)']
    # load physical proximity index from Pichler et al.
    FPI = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/epi/contacts/proximity/pichler_figure_S5_NACE64.csv'), index_col=[0])['physical_proximity_index']
    # multiply physical proximity and telework fraction and normalize --> hesitancy towards absenteism
    hesitancy = (FPI*f_remote) / sum(FPI*f_remote*(f_employees/sum(f_employees)))
    # load TDPF
    if ((scenarios == False) | (scenarios == 'hypothetical_spatial_spread')):
        if country == 'BE':
            labor_supply_shock_function = make_labor_supply_shock_function(IC_multiplier, country, age_classes, lmc_strateco, f_remote, f_workplace, hesitancy, simulation_start).get_economic_policy_BE
        else:
            labor_supply_shock_function = make_labor_supply_shock_function(IC_multiplier, country, age_classes, lmc_strateco, f_remote, f_workplace, hesitancy, simulation_start).get_economic_policy_SWE
    else:
        if country == 'BE':
            labor_supply_shock_function = make_labor_supply_shock_function(IC_multiplier, country, age_classes, lmc_strateco, f_remote, f_workplace, hesitancy, simulation_start).get_economic_policy_BE_scenarios

    # construct household demand shock TDPF (economic)
    # ================================================

    # load and the association vector between leisure and reduction in household demand (lav_leisure)
    lav_consumption = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/epi/contacts/proximity/leisure_association_vectors.csv'), index_col=[0])['consumption']
    # load demography per spatial patch
    demography = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).groupby(by='spatial_unit').sum().squeeze()
    if not spatial:
        demography = np.array([1,], dtype=float)
    else:
        demography = demography.values/sum(demography.values)
    # load TDPF
    from pyIEEM.models.TDPF import make_household_demand_shock_function
    household_demand_shock_function = make_household_demand_shock_function(IC_multiplier, country, lav_consumption, demography, simulation_start).get_household_demand_reduction
    
    # construct other demand shock TDPF (economic)
    # ============================================

    # get other accounts data
    d = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/eco/national_accounts/{country}/other_accounts_{country}_NACE64.csv'), index_col=0, header=0)
    # get right currency
    if country == 'SWE':
        curr = '(Mkr/y)'
        shock_investment = 0.0689 # Q2 2020, obtained from `DP_LIVE_16082023121712365.csv`
        shock_exports_goods = 0.14 # obtained from ``
        shock_exports_services = 0.21
    else:
        curr = '(M€/y)'
        shock_investment = 0.1617 # Q2 2020, obtained from `DP_LIVE_16082023121712365.csv`
        shock_exports_goods = 0.25 # obtained from `COMEXT_17082023124307665.csv`
        shock_exports_services = 0.21
    # add to dictionary
    parameters.update({
       'shock_investment': shock_investment,
       'shock_exports_goods': shock_exports_goods,
       'shock_exports_services': shock_exports_services,
    })
    # get total demand and all its core components except inventories
    total = d['Total other demand '+curr]
    IZW_government =  d['Other consumption - IZW '+curr] + d['Other consumption - government '+curr]
    exports = d['Other consumption - exports '+curr]
    investments = d['Other consumption - investments '+curr]
    # split exports of goods (A-F) and services (G-T) as these recover differently
    exports_goods = pd.Series(0, index=exports.index, name='exports_goods')
    exports_goods.loc[slice('A01','F41-43')] = exports.loc[slice('A01','F41-43')].values
    exports_services = pd.Series(0, index=exports.index, name='exports_goods')
    exports_services.loc[slice('G45',None)] = exports.loc[slice('G45',None)].values
    # initialize TDPF
    from pyIEEM.models.TDPF import make_other_demand_shock_function
    other_demand_shock_function = make_other_demand_shock_function(total, IC_multiplier, IZW_government, investments, exports_goods, exports_services, lav_consumption, demography, simulation_start).get_other_demand_reduction

    # initialize model
    # ================

    time_dependent_parameters = {
        'N': social_contact_function,
        'beta': seasonality_function,
        'kappa_S': labor_supply_shock_function,
        'kappa_D': household_demand_shock_function,
        'kappa_F': other_demand_shock_function,
    }

    model = epinomic_model(initial_states, parameters, coordinates=coordinates, time_dependent_parameters=time_dependent_parameters)

    return model

###############################
## Initialise epidemic model ##
###############################

def initialize_epidemic_model(country, age_classes, spatial, simulation_start, contact_type='absolute_contacts'):

    # get model parameters
    # ====================

    initial_states, parameters, coordinates = get_epi_params(country, age_classes, spatial, contact_type)

    # get calibrated initial states
    # =============================

    sim = xr.open_dataset(os.path.join(abs_dir, f'../../../data/interim/epi/initial_condition/{country}_INITIAL_CONDITION.nc'))
    initial_states={}
    for data_var in sim.keys():
        if spatial == True:
            initial_states.update({data_var: sim.sel(date=simulation_start)[data_var].values})   
        else:
            initial_states.update({data_var: np.expand_dims(sim.sum(dim='spatial_unit').sel(date=simulation_start)[data_var].values, axis=1)})   

    # construct social contact TDPF
    # =============================

    # get all necessary parameters
    parameters, demography, contacts, lmc_stratspace, lmc_strateco, f_workplace, f_remote, hesitancy, lav, f_employees, convmat = get_social_contact_function_parameters(parameters, country, spatial)
    # define all relevant parameters of the social contact function TDPF here
    parameters.update({'l': 7, 'mu': 1, 'nu': 24, 'xi_work': 5, 'xi_eff': 0.50, 'xi_leisure': 5,
                        'pi_work': 0.02, 'pi_eff': 0.06, 'pi_leisure': 0.30})
    # make social contact function
    from pyIEEM.models.TDPF import make_social_contact_function
    social_contact_function = make_social_contact_function(age_classes, demography, contact_type, contacts, lmc_stratspace, lmc_strateco, f_workplace, f_remote, hesitancy, lav,
                                                            False, f_employees, convmat, simulation_start, country)
    if country == 'SWE':
        social_contact_function = social_contact_function.get_contacts_SWE
    else:
        social_contact_function = social_contact_function.get_contacts_BE

    # construct seasonality TDPF
    # ==========================

    from pyIEEM.models.TDPF import make_seasonality_function
    seasonality_function = make_seasonality_function()
    
    if country == 'SWE':
        parameters.update({'amplitude': 0.20, 'peak_shift': 7})
    else: 
        parameters.update({'amplitude': 0.20, 'peak_shift': -14})    

    # initialize model
    # ================

    model = epidemic_model(initial_states, parameters, coordinates=coordinates, time_dependent_parameters={'N': social_contact_function, 'beta': seasonality_function})

    return model

######################
## helper functions ##
######################

def get_eco_params(country, prodfunc):
    """
    A function to load the economic model's parameters (excluding time-dependent parameters), initial states and coordinates

    input
    =====   

    country: str
        'BE' or 'SWE'

    prodfunc: str
        'leontief'/'strongly_critical'/'half_critical'/'weakly_critical'/'linear'

    output
    ======

    initial_states: dict
        Dictionary containing the non-zero initial values of the economic model

    parameters: dict
        Dictionary containing the (non time-dependent) parameters of the economic model

    coordinates: dict
        Dictionary containing the dimension names and corresponding coordinates of the economic model
    """

    # parameters
    # ==========

    ## Initialize parameters dictionary
    parameters = {}

    ## Input-Ouput matrix
    df = pd.read_csv(os.path.join(abs_dir, f"../../../data/interim/eco/national_accounts/{country}/IO_{country}_NACE64.csv"), sep=',',header=[0],index_col=[0])
    IO = df.values/365
    # others.csv
    df = pd.read_csv(os.path.join(abs_dir, f"../../../data/interim/eco/national_accounts/{country}/other_accounts_{country}_NACE64.csv"), sep=',',header=[0],index_col=[0])
    if country == 'SWE':
        curr = '(Mkr/y)'
    else:
        curr = '(M€/y)'

    ## National accounts
    parameters['x_0'] = np.array(df['Sectoral output ' + curr].values)/365
    parameters['c_0'] = np.array(df['Household demand ' + curr].values)/365
    parameters['f_0'] = np.array(df['Total other demand ' + curr].values)/365
    parameters['l_0'] = np.array(df['Labor compensation ' + curr].values)/365
    O_j = np.array(df['Intermediate demand ' + curr].values)/365

    ## Pichler et al.
    # desired stock
    df = pd.read_csv(os.path.join(abs_dir, f"../../../data/interim/eco/pichler/desired_stock_NACE64.csv"), sep=',',header=[0],index_col=[0])
    n = np.expand_dims(np.array(df['Desired stock (days)'].values), axis=1)
    # critical inputs
    df = pd.read_csv(os.path.join(abs_dir, f"../../../data/interim/eco/pichler/IHS_critical_NACE64.csv"), sep=',',header=[0],index_col=[0])
    parameters['C'] = df.values

    ## Computed variables

    # matrix of technical coefficients
    A = np.zeros([IO.shape[0],IO.shape[0]])
    for i in range(IO.shape[0]):
        for j in range(IO.shape[0]):
            A[i,j] = IO[i,j]/parameters['x_0'][j]
    parameters['A'] = A

    # Stock matrix under business as usual
    S_0 = np.zeros([IO.shape[0],IO.shape[0]])
    for i in range(IO.shape[0]):
        for j in range(IO.shape[0]):
            S_0[i,j] = IO[i,j]*n[j]
    parameters['St_0'] = S_0

    ## Hardcoded model parameters
    parameters.update({'delta_S': 0.75,                                                                                                                                                   
                       'eta': 14,                                                                                                 
                       'iota_H': 7,
                       'iota_F': 7,
                       'prodfunc': prodfunc,
                      })  

    ## Parameters that will be varied over time
    parameters.update({'kappa_S': np.zeros(63, dtype=float),
                       'kappa_D': np.zeros(63, dtype=float),
                       'kappa_F': np.zeros(63, dtype=float)})

    # coordinates
    # ===========

    coordinates = {'NACE64': df.index.values, 'NACE64_star': df.index.values}

    # initial states
    # ==============

    initial_states = {'x':parameters['x_0'],
                     'c': parameters['c_0'],
                     'f': parameters['f_0'],
                     'd': parameters['x_0'],
                     'l': parameters['l_0'],
                     'O': O_j,
                     'St': parameters['St_0']}

    return initial_states, parameters, coordinates

def get_epi_params(country, age_classes, spatial, contact_type):
    """
    A function to load the epidemiological model's parameters (excluding time-dependent parameters), initial states and coordinates

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
    N_home = contacts.loc['home', 'A', 'average', False,
                           slice(None), slice(None)][contact_type]
    
    N_other = N_home.copy(deep=True)
    for location in ['leisure_public', 'leisure_private', 'school']:
        N_other += contacts.loc[location, 'A', 'average', False,
                                slice(None), slice(None)][contact_type].values
    # convert to right demography
    N_home = aggregate_contact_matrix(N_home, age_classes, pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).groupby(by=['age']).sum().squeeze())

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
        # home contacts to np.array
        N_home = np.tile(np.expand_dims(N_home.values.reshape(
            2*[len(age_classes),]), axis=2), len(spatial_units))
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
        # home contacts to np.array
        N_home = np.expand_dims(
            N_home.values.reshape(2*[len(age_classes),]), axis=2)   
        # other contacts to np.array
        N_other = np.expand_dims(
            N_other.values.reshape(2*[len(age_classes),]), axis=2)

    # disease parameters
    # ==================

    # infectivity (R0 = 3.0)
    if country == 'BE':
        parameters = {'beta': 0.0335}
    else:
        parameters = {'beta': 0.0306} 

    # durations
    parameters.update({'alpha': 4.5,
                      'gamma': 0.7,
                      'delta': 5,
                      'epsilon': 14,
                      'zeta': 365/2, 
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
        'N': {'home': N_home, 'other': N_other, 'work': N_work},
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


def get_social_contact_function_parameters(parameters, country, spatial, scenarios):
    """
    A function to load, format and return all parameters necessary to construct the epidemiological model's time-dependent social contact function
    """

    # load NACE 21 composition per spatial patch
    if spatial == True:
        lmc_stratspace = pd.read_csv(os.path.join(
            abs_dir, f'../../../data/interim/eco/labor_market_composition/sector_structure_by_work_{country}.csv'), index_col=[0, 1])['rel'].sort_index()
    else:
        lmc_stratspace = pd.read_csv(os.path.join(
            abs_dir, f'../../../data/interim/eco/labor_market_composition/sector_structure_by_work_{country}.csv'), index_col=[0, 1])['abs']
        lmc_stratspace = lmc_stratspace.groupby(by='economic_activity').sum()/lmc_stratspace.groupby(by='economic_activity').sum().sum()
    
    # load the number of employees in every sector of the NACE 64 from the national accounts
    f_employees = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/eco/national_accounts/{country}/other_accounts_{country}_NACE64.csv'), index_col=[0])['Number of employees (-)']

    ## lmc_strateco
    # get labor market composition (abs)
    df = pd.read_csv(os.path.join(abs_dir, f'../../../data/interim/eco/labor_market_composition/sector_structure_by_work_{country}.csv'), index_col=[0, 1], header=0)['abs'].sort_index()
    # convert to the fraction of laborers in spatial patch 'i' working in sector 'X' of the total number of laborers working in sector 'X' (NACE 21)
    lmc_strateco = df/df.groupby('economic_activity').transform('sum')
    # convert from NACE 21 to NACE 64 using the ratios found in the national accounts
    iterables = [lmc_strateco.index.get_level_values('spatial_unit').unique().values, f_employees.index]
    names = ['spatial_unit', 'economic_activity']
    out = pd.Series(index=pd.MultiIndex.from_product(iterables, names=names), name='f_employees', dtype=float)
    for act in f_employees.index.values:
        out.loc[slice(None), act] = lmc_strateco.loc[slice(None), act[0]].values # *f_employees.loc[act]
    lmc_strateco = out

    # load social contacts
    contacts = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/epi/contacts/matrices/FR/comesf_formatted_matrices.csv'), index_col=[0, 1, 2, 3, 4, 5],
                           converters={'age_x': to_pd_interval, 'age_y': to_pd_interval})

    # load national demography
    demography = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).groupby(by='age').sum().squeeze()

    # load fraction employees in workplace during pandemic
    f_workplace = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/epi/contacts/proximity/ermg_summary.csv'), index_col=[0])['workplace']

    # load telework fraction observed during pandemic
    f_remote = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/epi/contacts/proximity/ermg_summary.csv'), index_col=[0])['remote']

    # load and normalise leisure association vector (lav)
    lav = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/epi/contacts/proximity/leisure_association_vectors.csv'), index_col=[0])['contacts']
    lav = lav/sum(lav)

    # load physical proximity index from Pichler et al.
    FPI = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/epi/contacts/proximity/pichler_figure_S5_NACE64.csv'), index_col=[0])['physical_proximity_index']

    # multiply physical proximity and telework fraction and normalize --> hesitancy towards absenteism
    hesitancy = (FPI*f_remote) / sum(FPI*f_remote*(f_employees/sum(f_employees)))

    # compute fraction of employees in NACE 64 sector as a percentage of its NACE 21 sector
    f_employees = f_employees.reset_index()
    f_employees['NACE 21'] = f_employees['index'].str[0]
    f_employees = f_employees.rename(columns={'index': 'NACE 64'})
    f_employees = f_employees.groupby(['NACE 21', 'NACE 64'])['Number of employees (-)'].sum().reset_index()
    f_employees['fraction_NACE21'] = f_employees['Number of employees (-)'] / f_employees.groupby('NACE 21')['Number of employees (-)'].transform('sum')
    f_employees = f_employees.drop(columns = ['NACE 21', 'Number of employees (-)']).set_index('NACE 64').squeeze()
    
    # NACE 64 to NACE 21 conversion matrix
    convmat = pd.read_csv(os.path.join(abs_dir, f'../../../data/interim/eco/misc/conversion_matrix_NACE64_NACE21.csv'), index_col=[0], header=[0])
    convmat = convmat.fillna(0).values

    # define economic policies
    if ((scenarios == False) | (scenarios == 'hypothetical_spatial_spread')):
        # load economic policies
        policies_df = pd.read_csv(os.path.join(abs_dir, f'../../../data/interim/eco/policies/policies_{country}.csv'), index_col=[0], header=[0])
        # extract and format
        if country == 'BE':
            parameters.update({'economy_BE_lockdown_1': np.expand_dims(policies_df['lockdown_1'].values, axis=1),
                                'economy_BE_phaseI': np.expand_dims(policies_df['lockdown_release_phaseI'].values, axis=1),
                                'economy_BE_phaseII': np.expand_dims(policies_df['lockdown_release_phaseII'].values, axis=1),
                                'economy_BE_phaseIII': np.expand_dims(policies_df['lockdown_release_phaseIII'].values, axis=1),
                                'economy_BE_phaseIV': np.expand_dims(policies_df['lockdown_release_phaseIV'].values, axis=1),
                                'economy_BE_lockdown_Antwerp': np.expand_dims(policies_df['lockdown_Antwerp'].values, axis=1),
                                'economy_BE_lockdown_2_1': np.expand_dims(policies_df['lockdown_2_1'].values, axis=1),
                                'economy_BE_lockdown_2_2': np.expand_dims(policies_df['lockdown_2_2'].values, axis=1),
                                'economy_BE_plateau': np.expand_dims(policies_df['lockdown_plateau'].values, axis=1),
                                })
        else:
            parameters.update({'economy_SWE': np.expand_dims(policies_df['policy'].values, axis=1)})
    else:
        if country == 'BE':
            # load economic policies
            policies_df = pd.read_csv(os.path.join(abs_dir, f'../../../data/interim/eco/policies/policies_{country}_scenarios.csv'), index_col=[0], header=[0])
            parameters.update({'L1': np.expand_dims(policies_df['L1'].values, axis=1),
                               'L2_schools': np.expand_dims(policies_df['L2_schools'].values, axis=1),
                               'L2': np.expand_dims(policies_df['L2'].values, axis=1),
                               'L3_schools': np.expand_dims(policies_df['L3_schools'].values, axis=1),
                               'L3': np.expand_dims(policies_df['L3'].values, axis=1),
                               'L4': np.expand_dims(policies_df['L4'].values, axis=1),
                               't_start_lockdown': datetime(2020, 3, 15),
                               't_end_lockdown': datetime(2020, 5, 4),
                               'scenario': 'L1',
                               'l_release': 31,
                               })

    return parameters, demography, contacts, lmc_stratspace, lmc_strateco, f_workplace, f_remote, hesitancy, lav, f_employees, convmat

################################################
## Aggregation functions Brussels and Brabant ##
################################################

import xarray as xr
def aggregate_Brussels_Brabant_Dataset(simulation_in):
    """
    A wrapper for `aggregate_Brussels_Brabant()`, converting all model states into the aggregated format

    Input
    =====
    
    simulation_in: xarray.Dataset
        Simulation result (arrondissement or provincial level)
    
    Output
    ======
    
    simulation_out: xarray.Dataset
        Simulation result. Provincial spatial aggregation with Bruxelles and Brabant aggregated into NIS 21000
    """
    output = []
    for state in simulation_in.keys():
        o = aggregate_Brussels_Brabant_DataArray(simulation_in[state])
        o.name = state
        output.append(o)
    return xr.merge(output)

def dummy_aggregation(simulation_in):
    return simulation_in

def aggregate_Brussels_Brabant_DataArray(simulation_in):
    """
    A function to aggregate an arrondissement simulation to the provincial level.
    A function to aggregate the provinces of Brussels, Brabant Wallon and Vlaams Brabant into one province.
    
    Input
    =====
    
    simulation_in: xarray.DataArray
        Simulation result (arrondissement or provincial level)
    
    Output
    ======
    
    simulation_out: xarray.DataArray
        Simulation result. Provincial spatial aggregation with Bruxelles and Brabant aggregated into NIS 21000
    """

    # define new names
    new_names = ['Antwerpen', 'Brussels and Brabant', 'Hainaut', 'Liege', 'Limburg', 'Luxembourg', 'Namur', 'Oost-Vlaanderen', 'West-Vlaanderen']
    # preallocate tensor for the converted output
    if 'draws' in simulation_in.dims:
        data = np.zeros([len(new_names),
                        len(simulation_in.coords['draws']),
                        len(simulation_in.coords['date']),
                        len(simulation_in.coords['age_class'])])
    else:
        data = np.zeros([len(new_names),
                        len(simulation_in.coords['date']),
                        len(simulation_in.coords['age_class'])])
    # aggregate Brussels and Brabant
    for i, prov in enumerate(new_names):
        if prov != 'Brussels and Brabant':
            data[i,...] = simulation_in.sel(spatial_unit=prov).values
        else:
            data[i,...] = simulation_in.sel(spatial_unit='Brussels').values + simulation_in.sel(spatial_unit='Vlaams-Brabant').values + \
                            simulation_in.sel(spatial_unit='Brabant Wallon').values        
    # Send to simulation out
    if 'draws' in simulation_in.dims:
        data=np.swapaxes(np.swapaxes(np.swapaxes(data,0,1), 1,2), 2,3)
        coords=dict(draws = simulation_in.coords['draws'],
                    date = simulation_in.coords['date'],
                    age_class = simulation_in.coords['age_class'],
                    spatial_unit=(['spatial_unit'], new_names),
                    )
    else:
        data=np.swapaxes(np.swapaxes(data,0,1), 1, 2)
        coords=dict(date = simulation_in.coords['date'],
                    spatial_unit=(['spatial_unit'], new_names),
                    age_class = simulation_in.coords['age_class'],
            )
    return xr.DataArray(data, dims=simulation_in.dims, coords=coords)

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
def is_school_holiday(d, country):
    """
    A function returning 'True' if a given date is a school holiday or primary and secundary schools in Belgium or Sweden.
    Tertiary education is not considered in this work.
    
    Main differences BE and SWE:
        - Summer holiday. SWE: mid Jun - mid Aug. BE: Jul - Aug.
        - Easter holiday. SWE: Good Friday + Easter Monday. BE: Two weeks holiday.

    Input
    =====
    
    d: datetime.datetime
        Current simulation date

    country: str
        'BE' or 'SWE'

    Returns
    =======
    
    is_school_holiday: bool
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
    if country == 'BE':
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
        d_easter - timedelta(days=2),                # Good Friday
        d_easter + timedelta(days=1),                # Easter
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
    if country == 'BE':
        if ((d.isocalendar().week in holiday_weeks) | \
                (d in public_holidays)) | \
                    ((datetime(year=d.year, month=7, day=1) <= d < datetime(year=d.year, month=9, day=1))):
            return True
        else:
            return False
    else:
        # Summer holiday is shifted two weaks in Sweden
        if ((d.isocalendar().week in holiday_weeks) | \
                (d in public_holidays)) | \
                    ((datetime(year=d.year, month=6, day=15) <= d < datetime(year=d.year, month=9, day=1))):
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