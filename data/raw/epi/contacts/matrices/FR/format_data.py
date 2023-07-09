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
        key = (entry[0], entry[1], frozenset(entry[2]), frozenset(entry[3]))
        if key not in unique_tuples:
            unique_tuples.add(key)
            result.append(entry)
    return result

# Define desired age groups of final matrices
age_classes = pd.IntervalIndex.from_tuples([(0,20),(20,40),(40,60),(60,80),(80,120)], closed='left')

# Define names for locations, durations and professions of interest
locations = ['home', 'school', 'work_indoor', 'leisure_private', 'leisure_public', 'transport', 'work_leisure_outdoor']
durations = [2.5, 10, 37.5, 150, 240]

# Define output dataframe for all contacts except work
names = ['location', 'duration', 'age_x', 'age_y']
iterables = [locations, durations, age_classes, age_classes]
df = pd.DataFrame(0, index=pd.MultiIndex.from_product(iterables, names=names), columns=["number_contacts", "number_survey_participants"])

# Load dataset
data = pd.read_excel(os.path.join(abs_dir, 'RawData_ComesF_extract.xlsx'), sheet_name="CONTACT", header=[0,1,2])

# Drop columns containing NBcontacts
data.sort_index(axis=1).drop(columns=['NBcontact1'], inplace=True)

dropped_count=0
# Loop over the survey participants
for i in tqdm(range(len(data))):

    # Determine the personal characteristics
    row = data.iloc[i]
    age = row['Q1']['Q3_1']["Age du sujet de l'enquête"]
    age_x = age_classes[age_classes.contains(age)]

    # Loop over the contacts
    participant_count=[]
    for j in range(1,69):
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
            df.loc[(location, duration, age_x, age_y), 'number_contacts'] += 1
            # Keep track of the participants data
            participant_count.append((location, duration, age_x, age_y),)

    # Drop the duplicate data
    for index in drop_duplicates(participant_count):
        df.loc[index, 'number_survey_participants'] += 1

# Print the fraction of dropped contacts
print(f"\n{100*dropped_count/(len(data)*69):.1f} % of the {len(data)*69} reported contacts were dropped because the age, duration or location was missing\n")

# Save the resulting dataframe in a .csv
df.to_csv('FormatData_ComesF.csv')