""" A script to format the raw dataset by CoMEs-F in a more pleasant and easy-to-use format
"""

import os
import sys
import math
import numpy as np
import pandas as pd 
from tqdm import tqdm

# Define absolute paths only
abs_dir = os.path.dirname(__file__)

# Helper function
def drop_duplicates(input_list):
    unique_tuples = set()
    result = []
    for entry in input_list:
        key = (entry[0], entry[1], entry[2], entry[3], frozenset(entry[4]), frozenset(entry[5]))
        if key not in unique_tuples:
            unique_tuples.add(key)
            result.append(entry)
    return result

# Define desired age groups of final matrices
age_classes = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left')

# Define names for locations, durations and professions of interest
locations = ['home', 'school', 'work_indoor', 'leisure_private', 'leisure_public', 'transport', 'work_leisure_outdoor']
durations = ['< 5 min', '5-15 min', '15-60 min', '60-240 min', '> 240 min']
daytypes = ['weekday', 'weekendday']
vacations = [True, False]

# Define output dataframe for all contacts except work
names = ['location', 'duration', 'type_day', 'vacation', 'age_x', 'age_y']
iterables = [locations, durations, daytypes, vacations, age_classes, age_classes]
df = pd.DataFrame(0, index=pd.MultiIndex.from_product(iterables, names=names), columns=["number_contacts", "number_survey_participants"])

# Load dataset
data = pd.read_excel(os.path.join(abs_dir, 'RawData_ComesF.xlsx'), sheet_name="CONTACT", header=[0,1,2])
data.sort_index(axis=1).drop(columns=['NBcontact1'], inplace=True)

dropped_count=0
# SURVEY PARTICIPANT LOOP
for i in tqdm(range(len(data))):
    # Determine the personal characteristics
    row = data.iloc[i]
    age = row['Q1']['Q3_1']["Age du sujet de l'enquête"]
    age_x = age_classes[age_classes.contains(age)]
    # Extract the daytypes
    days_of_week = row.loc['Jour1'].droplevel(0).values
    # Vacation yes or no?
    is_vacation = row.loc['VAC1'].droplevel(0).values
    # Perform checks on impossible day types
    if ((any(math.isnan(d) for d in days_of_week)) | (any(d > 7 for d in days_of_week)) | (any(d < 1 for d in days_of_week))):
        dropped_count+=1
    else:
        # CONTACT LOOP
        participant_count=[]
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
            # If age of contact is missing this means no contact was made
            if not math.isnan(contact_data.loc['age moyen ']):
                # Check if age, duration and location of known, drop the contact if this is not the case
                if ((math.isnan(contact_data.loc['age moyen '])) | (math.isnan(contact_data.loc['durée '])) | (all(v == 0 for v in contact_data.iloc[4:11].values))):
                    dropped_count+=1
                else:
                    # Extract age of contacted person and duration
                    age_y = age_classes[age_classes.contains(contact_data.loc['age moyen '])]
                    duration = durations[int(contact_data.loc['durée ']-1)]
                    # Extract location, keep track of dropped contacts due to missing location
                    loc = contact_data.iloc[4:11].values
                    loc = loc!= 0
                    location = locations[np.where(loc)[0][0]]
                # Assign data
                df.loc[(location, duration, daytype, vacation, age_x, age_y), 'number_contacts'] += 1
                # Keep track of the participants data
                participant_count.append((location, duration, daytype, vacation, age_x, age_y),)
        # Drop the duplicate data
        for index in drop_duplicates(participant_count):
            df.loc[index, 'number_survey_participants'] += 1

# Print the fraction of dropped contacts
print(f"\n{100*dropped_count/(len(data)*69):.1f} % of the {len(data)*69} reported contacts were dropped because the age, duration, daytype, or location was missing\n")

print(df.loc['home', '15-60 min', 'weekday', True, slice(None), slice(None)])
print(df.loc['home', '15-60 min', 'weekday', False, slice(None), slice(None)])
print(df.loc['home', '15-60 min', 'weekendday', True, slice(None), slice(None)])

# Save the resulting dataframe in a .csv
df.to_csv('FormatData_ComesF.csv')