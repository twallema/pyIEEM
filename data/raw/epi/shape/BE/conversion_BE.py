"""Cleans the geojson file for Sweden
"""

import os
import pandas as pd
import geopandas as gpd

abs_dir = os.path.dirname(__file__)

# Load geojson file
BE = gpd.read_file(os.path.join(abs_dir, 'BEL_adm2.shp'))

# Merge Limburg
limburg = BE.loc[BE.index[[2,9]]]
limburg = limburg.dissolve(by='NAME_2', aggfunc='sum')

# Drop first Limburg from original dataframe
BE.drop(2, inplace=True)

# Define a better index name
BE['simple_names'] = ['Brussel', 'Antwerpen', 'Oost-Vlaanderen', 'Vlaams Brabant', 'West-Vlaanderen', 'Brabant Wallon',
                        'Hainaut', 'Liege', 'Limburg', 'Luxembourg', 'Namur']
BE.set_index('simple_names', inplace=True)
BE.index.name = 'spatial_unit'

# Drop all columns except geometry
BE.drop(columns=[x for x in list(BE.columns) if x != 'geometry'], inplace=True)

# Add merged limburg
BE.loc['Limburg'] = limburg['geometry'].values[0]

# Convert to cartesian coordinates
BE = BE.to_crs({'init': 'epsg:3857'})

# Compute area in km2
BE["area"] = BE['geometry'].area/10**6

# Attach demographic data (2019, retrieved from StatBel)
BE['pop'] = [1208542, 1857986, 1515064, 1146175, 1195796, 403599, 1344241, 1106992, 874048, 284638, 494325]

# Compute population density
BE['pop_density'] = BE['pop']/BE['area']

# Save result
BE.to_file(os.path.join(abs_dir, '../../../../interim/epi/shape/BE.json'), driver="GeoJSON")