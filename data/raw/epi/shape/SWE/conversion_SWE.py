"""Cleans the geojson file for Sweden
"""

import os
import pandas as pd
import geopandas as gpd

abs_dir = os.path.dirname(__file__)

# Load geojson file
SWE = gpd.read_file(os.path.join(abs_dir, 'SWE.json'))
# Convert to cartesian coordinates
SWE = SWE.to_crs({'init': 'epsg:3857'})
# Define a better index name
SWE['spatial_unit'] = ['Ostergotland', 'Blekinge', 'Dalarna', 'Gavleborg', 'Gotland', 'Halland', 'Jamtland', 'Jonkoping', 'Kalmar',
                            'Kronoberg', 'Norrbotten', 'Orebro', 'Sodermanland', 'Skane', 'Stockholm', 'Uppsala', 'Varmland', 'Vasterbotten', 'Vasternorrland',
                                'Vastmanland', 'Vastra Gotaland']
#SWE.set_index('spatial_unit', inplace=True)
#SWE.index.name = 'spatial_unit'
# Drop all columns except geometry
SWE.drop(columns=[x for x in list(SWE.columns) if ((x != 'geometry')&(x != 'spatial_unit'))], inplace=True)
# Add demography
dem = pd.read_csv(os.path.join(abs_dir, '../../../../interim/epi/demographic/population_density_SWE_2019.csv'), index_col=0)
SWE = SWE.merge(dem, on='spatial_unit')
# Save result
SWE.to_file(os.path.join(abs_dir, '../../../../interim/epi/shape/SWE.json'), driver="GeoJSON")