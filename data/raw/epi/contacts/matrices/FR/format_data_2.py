""" A script to format the raw dataset by CoMEs-F in a more pleasant and easy-to-use format
"""

import os
import sys
import math
import numpy as np
import pandas as pd 
from tqdm import tqdm

# Helper function
def drop_duplicates(input_list):
    unique_tuples = {}
    result = []
    duplicates_count = []
    for entry in input_list:
        key = (entry[0], frozenset(entry[1]), entry[2], frozenset(entry[3]), entry[4], entry[5], entry[6], entry[7])
        if key not in unique_tuples:
            unique_tuples[key] = 1
            result.append(entry)
        else:
            unique_tuples[key] += 1
    for value in unique_tuples.values():
        duplicates_count.append(value)
    return result, duplicates_count

# Define absolute paths only
abs_dir = os.path.dirname(__file__)

# Define desired age groups of final matrices
age_classes = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left')

# Load dataset
data = pd.read_excel(os.path.join(abs_dir, 'RawData_ComesF.xlsx'), sheet_name="CONTACT", header=[0,1,2])
data.sort_index(axis=1).drop(columns=['NBcontact1'], inplace=True)


# Translation
translations = {
    'sector': ['A', 'C10-12', 'C', 'D', 'F', 'G', 'K', 'M', 'R, S, T', 'P, Q', 'O, N'],
    'location': ['home', 'school', 'work_indoor', 'leisure_private', 'leisure_public', 'transport', 'work_leisure_outdoor'],
    'duration': ['< 5 min', '5-15 min', '15-60 min', '60-240 min', '> 240 min'],
    'age_y': [str(a) for a in age_classes.values],
    'daytype': ['weekendday', 'weekday'],
    'vacation': [True, False]
}

# Pre-allocate dictionary for the output
columns = ['ID', 'age_x', 'sector', 'age_y', 'location', 'duration', 'daytype', 'vacation', 'reported_contacts']
output = {k: [] for k in columns}

dropped_count=0
for i in tqdm(range(len(data))):
    # Assign ID to survey participant
    ID = i
    # Personal characteristics: age group
    row = data.iloc[i]
    age = row['Q1']['Q3_1']["Age du sujet de l'enquête"]
    age_x = age_classes[age_classes.contains(age)].values
    age_x = age_x.astype(str)[0]
    # Personal characteristics: sector of employment
    sector_n = row['Q10']['Q7']["Dans quel secteur d'activité travaillez-vous"]
    if ((not math.isnan(sector_n)) & (sector_n >= 1) & (sector_n <= 11)):
        sector = translations['sector'][int(sector_n - 1)]
    else:
        sector = 'NA'
    # Temporal characteristics: type of day
    days_of_week = row.loc['Jour1'].droplevel(0).values
    # Temporal characteristics: vacation or not
    is_vacation = row.loc['VAC1'].droplevel(0).values
    # Perform checks on impossible day types
    if ((any(math.isnan(d) for d in days_of_week)) | (any(d > 7 for d in days_of_week)) | (any(d < 1 for d in days_of_week))):
        dropped_count+=1
    else:
        # CONTACT LOOP
        contact_properties=[]
        for j in range(1,69):
            # Extract correct daytype/vacation
            if j <= 40:
                vacation = is_vacation[0]
                if vacation == 1:
                    vacation = False
                elif vacation == 2:
                    vacation = True
                daytype = days_of_week[0]
                if daytype <= 5:
                    daytype = 'weekday'
                else:
                    daytype = 'weekendday'
            else:
                vacation = is_vacation[1]
                if vacation == 1:
                    vacation = False
                elif vacation == 2:
                    vacation = True
                daytype = days_of_week[1]
                if daytype <= 5:
                    daytype = 'weekday'
                else:
                    daytype = 'weekendday'
            # Extract the contact's data
            contact_data = row.loc[f'contact  {j}'].droplevel(0)
            if not math.isnan(contact_data.loc['age moyen ']):
                # Check if age, duration and location are known, drop the contact if this is not the case
                if ((math.isnan(contact_data.loc['age moyen '])) | (math.isnan(contact_data.loc['durée '])) | (all(v == 0 for v in contact_data.iloc[4:11].values))):
                    dropped_count+=1
                else:
                    # Extract age of contacted person and duration
                    age_y = age_classes[age_classes.contains(contact_data.loc['age moyen '])].values
                    age_y = age_y.astype(str)[0]
                    duration = translations['duration'][int(contact_data.loc['durée ']-1)]
                    # Extract location, keep track of dropped contacts due to missing location
                    loc = contact_data.iloc[4:11].values
                    loc = loc!= 0
                    location = translations['location'][np.where(loc)[0][0]]

                # Keep track of the contacts data so we can eliminate doubles
                contact_properties.append((ID, age_x, sector, age_y, location, duration, daytype, vacation),)

        # Drop the duplicate data
        unique_indices, contact_counts = drop_duplicates(contact_properties)

        # Add a count of zero for all age_y not present
        #age_y_present = [id[3] for id in unique_indices]
        #for item in translations['age_y']:
        #    if item not in age_y_present:
        #        unique_indices.extend([(ID, age_x, sector, item, location, duration, daytype, vacation),])
        #        contact_counts.extend([0,])

        # Append to output dictionary
        output['reported_contacts'].extend(contact_counts)
        for unique_index in unique_indices:
            for i, key in enumerate(output.keys()):
                if key != 'reported_contacts':
                    output[key].append(unique_index[i])

# Print the fraction of dropped contacts
print(f"\n{100*dropped_count/(len(data)*69):.1f} % of the {len(data)*69} reported contacts were dropped because the age, duration, daytype, or location was missing\n")

# Define output dataframe containing all contact-related characteristics in index and person-related characteristics in the columns
names = ['ID', 'age_y', 'location', 'duration', 'type_day', 'vacation']
iterables = [np.unique(output['ID']), translations['age_y'], translations['location'], translations['duration'], translations['daytype'], translations['vacation']]
df = pd.DataFrame(0, index=pd.MultiIndex.from_product(iterables, names=names), columns=["age_x", "sector", "reported_contacts"])

# Fill in correct age_x and sector for every ID
for id in tqdm(np.unique(output['ID'])):
    df.loc[df.index.get_level_values('ID') == id, 'age_x'] = str(np.unique(np.array(output['age_x'])[output['ID']==id])[0])
    df.loc[df.index.get_level_values('ID') == id, 'sector'] = str(np.unique(np.array(output['sector'])[output['ID']==id])[0])

# Fill in present values
for ID, age_x, sector, age_y, location, duration, daytype, vacation, contacts in tqdm(zip(output['ID'], output['age_x'], output['sector'], output['age_y'], output['location'], output['duration'], output['daytype'], output['vacation'], output['reported_contacts'])):
    df.loc[(ID, age_y, location, duration, daytype, vacation),'reported_contacts'] = contacts

# Save result
df.to_csv('FormatData_ComesF_2.csv')


