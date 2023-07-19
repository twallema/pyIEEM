import os
import numpy as np
import pandas as pd
abs_dir = os.path.dirname(__file__)

# Desired names
desired_names = ['Antwerpen', 'Vlaams-Brabant', 'Brabant Wallon', 'Brussels', 'West-Vlaanderen', 'Oost-Vlaanderen', 'Hainaut', 'Liege', 'Limburg', 'Luxembourg', 'Namur']
extract_names = ['Provincie Antwerpen', 'Provincie Vlaams-Brabant', 'Provincie Waals-Brabant', 'Arrondissement Brussel-Hoofdstad', 'Provincie West-Vlaanderen',
                    'Provincie Oost-Vlaanderen', 'Provincie Henegouwen', 'Provincie Luik', 'Provincie Limburg', 'Provincie Luxemburg', 'Provincie Namen']

# Get names
df = pd.read_excel(os.path.join(abs_dir, 'Pop_LPW_NL_25FEB15_delete_unknown.xlsx'), sheet_name="Tabel1_2011", header=5)
excel_names = df['Verblijfplaats'].dropna().values

# Extract matrix
df = pd.read_excel(os.path.join(abs_dir, 'Pop_LPW_NL_25FEB15_delete_unknown.xlsx'), sheet_name="Tabel1_2011")
rows=[]
working_pop=[]
for i in df['00.24 - Werkende bevolking volgens geslacht, verblijfplaats en plaats van tewerkstelling'].loc[5:1942].dropna().index:
    rows.append(df.iloc[i+2,4:-1].values)
    working_pop.append(df.iloc[i+2,4])
matrix = np.zeros([len(rows),len(rows)])
for j in range(len(rows)):
    matrix[j,:]=rows[j]
mobility_df = pd.DataFrame(matrix,index=excel_names,columns=excel_names)
working_pop_df = pd.Series(working_pop,index=excel_names)

# Extract provinces only and assign desired names
mobility_df=pd.DataFrame(mobility_df.loc[extract_names, extract_names].values, index=desired_names,columns=desired_names)
working_pop_df=pd.Series(working_pop_df.loc[extract_names].values, index=desired_names)

# Extract the number of employed people per province according to the census
n_employed = pd.read_csv(os.path.join(abs_dir, '../../../eco/labor_market_composition/active_population_BE.csv'), index_col=0)['population_15_64']* \
                pd.read_csv(os.path.join(abs_dir, '../../../eco/labor_market_composition/active_population_BE.csv'), index_col=0)['fraction_employed_15_64']       

# Recurrent mobility data is imperfect 
# -> Sum over rows should should match degrees of employment in every province (which it doesn't; see below)

# spatial_unit          # computed            # actual (2011 census)
#Antwerpen              0.63                  0.68
#Vlaams-Brabant         0.66                  0.69
#Brabant Wallon         0.61                  0.66
#Brussels               0.46                  0.57
#West-Vlaanderen        0.67                  0.70
#Oost-Vlaanderen        0.67                  0.70
#Hainaut                0.53                  0.62
#Liege                  0.55                  0.64
#Limburg                0.60                  0.67
#Luxembourg             0.46                  0.66
#Namur                  0.59                  0.66

# Compute fraction that is missing
missing_employees = n_employed - mobility_df.sum(axis=1)

# Compute ratios of travelers traveling to every other province
ratios = mobility_df/mobility_df.sum(axis=1)

# Distribute missing employees using the ratios in the census
for prov in ratios.index:
    mobility_df.loc[prov, :] += (ratios.loc[prov, :] * missing_employees.loc[prov]).values

# Extract total population
total_pop_df = pd.read_csv(os.path.join(abs_dir, '../../../eco/labor_market_composition/active_population_BE.csv'), index_col=0)['total_population']

# Perform row-wise division by total population
for i in range(len(desired_names)):
        mobility_df.values[i,:] = mobility_df.values[i,:]/total_pop_df.values[i]

# Save result
mobility_df=pd.DataFrame(mobility_df.values, index=desired_names,columns=desired_names)

# Save result
mobility_df.to_csv(os.path.join(abs_dir,'recurrent_mobility_BE.csv'))
