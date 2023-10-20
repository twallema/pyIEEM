import os
import numpy as np
import pandas as pd
from datetime import datetime

abs_dir = os.path.dirname(__file__)

def get_economic_data(indicator, country, relative=False):
    """
    A function to load and format the GDP or employment data of country `country`

    input
    =====

    indicator: str
        economic indicator. 'GDP' or 'employment'.
        
    country: str
        country name. 'BE' or 'SWE'

    relative: bool
        return relative data if True

    output
    ======

    data: pd.Series
        GDP or employment data
    """

    # input check
    assert ((indicator == 'GDP') | (indicator == 'employment')), f"invalid economic indicator '{indicator}'. valid indicators are 'GDP' and 'employment'"
    assert ((country=='BE') | (country=='SWE')), f"invalid country name '{country}'. valid names are 'BE' and 'SWE'."

    # load data
    data = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/eco/calibration_data/{country}.csv'), index_col=[0], parse_dates=True)[indicator].dropna()

    # convert to absolute data if needed
    if relative == False:
        other_accounts = pd.read_csv(os.path.join(abs_dir, f"../../../data/interim/eco/national_accounts/{country}/other_accounts_{country}_NACE64.csv"), sep=',',header=[0],index_col=[0])
        if country == 'SWE':
            curr = '(Mkr/y)'
        else:
            curr = '(M€/y)'

        if indicator == 'GDP':
            data *= sum(np.array(other_accounts['Sectoral output ' + curr].values)/365)
        else:
            data *= sum(np.array(other_accounts['Labor compensation ' + curr].values)/365)

    return data

def get_hospitalisation_incidence(country, aggregate_bxl_brabant=False):
    """
    A function to load and format the hospitalisation incidence data of country `country`

    input
    =====

    country: str
        country name. 'BE' or 'SWE'

    output
    ======

    data: pd.Series
        daily hospitalisation incidence

    """

    # input check
    assert ((country=='BE') | (country=='SWE')), f"invalid country name '{country}'. valid names are 'BE' and 'SWE'."

    # load data
    data = pd.read_csv(os.path.join(
        abs_dir, f'../../../data/interim/epi/cases/hospital_incidence_{country}.csv'), index_col=[0], parse_dates=True)

    if country == 'SWE':
        # format and sort alphabetically
        data = data.stack()
        data = data.rename('hospital incidence')
        data.index.names = ['date', 'spatial_unit']
        # data is weekly incidence --> divide by seven
        data /= 7
        # sort alphabetically
        data = data.sort_index()
    else:
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
        desired_data = pd.Series(index=pd.MultiIndex.from_product(iterables, names=names), name='hospital incidence', dtype=float).sort_index()     
        # slice right column and set the index
        data = data[['PROVINCE', 'NEW_IN']].groupby(by=['DATE', 'PROVINCE']).sum()
        data = data.squeeze().rename('hospital incidence')
        data.index.names = ['date', 'spatial_unit']
        # merge dataframes
        data = desired_data.combine_first(data).fillna(0)
        # aggregate brussels and both brabants if desired
        if aggregate_bxl_brabant:
            data.loc[slice(None), 'Brussels'] = (data.loc[slice(None), 'Brussels'] + data.loc[slice(None), 'Vlaams-Brabant'] + data.loc[slice(None), 'Brabant Wallon']).values
            data = data.reset_index()
            data = data[((data['spatial_unit'] != 'Vlaams-Brabant') & (data['spatial_unit'] != 'Brabant Wallon'))]
            data['spatial_unit'][data['spatial_unit'] == 'Brussels'] = 'Brussels and Brabant'
            data = data.groupby(by=['date','spatial_unit']).sum().squeeze()

    return data
