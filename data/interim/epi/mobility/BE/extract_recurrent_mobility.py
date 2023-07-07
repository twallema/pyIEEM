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

# Extract total population 18-60 yo
active_pop_df = pd.read_csv(os.path.join(abs_dir, 'active_population_2011_format.csv'), index_col=0)

# Perform row-wise division by total population
for i in range(len(desired_names)):
        mobility_df.values[i,:] = mobility_df.values[i,:]/active_pop_df.values[i]

print(f'\nTotal fraction of Belgian active population with a job: {np.sum(working_pop_df.values)/np.sum(active_pop_df.values)*100:.1f} %\n')

# Save result
mobility_df=pd.DataFrame(mobility_df.values, index=desired_names,columns=desired_names)

# Save result
mobility_df.to_csv(os.path.join(abs_dir,'recurrent_mobility_BE.csv'))
