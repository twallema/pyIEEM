import os
import pandas as pd
import geopandas as gpd

abs_dir = os.path.dirname(__file__)

# Load geojson files
SWE = gpd.read_file(os.path.join(abs_dir, '../data/interim/epi/shape/SWE.json'))
BE = gpd.read_file(os.path.join(abs_dir, '../data/interim/epi/shape/BE.json'))

# Load NACE 21 composition at provincial level

# Load contact matrices

# Compute total number of contacts per region

# Perhaps also append this to the geojson files

# Visualize result
