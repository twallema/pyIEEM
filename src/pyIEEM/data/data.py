import os
import pandas as pd
from datetime import datetime

abs_dir = os.path.dirname(__file__)

def get_hospitalisation_incidence(country):
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
        # divide number of hosp. in spatial patch 'Missing' over other spatial patches using fraction of hospitalisations occuring in every spatial patch
        fraction = data[data.columns[data.columns != 'Missing']].div(
            data[data.columns[data.columns != 'Missing']].sum(axis=1), axis=0)
        data = (data[data.columns[data.columns != 'Missing']] +
                fraction.multiply(data['Missing'].values, axis=0)).fillna(0)
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
        data = data.loc[slice(None, datetime(2021, 1, 1)), :]
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

    return data
