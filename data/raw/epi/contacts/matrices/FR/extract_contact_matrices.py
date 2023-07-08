import os
import sys
import math
import numpy as np
import pandas as pd 

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

# Define locations, durations and professions of interest
locations = ['home', 'school', 'work_indoor', 'leisure_private', 'leisure_public', 'transport', 'work_leisure_outdoor']
durations = [2.5, 10, 37.5, 150, 240]

# Define output dataframe for all contacts except work
names = ['location', 'duration', 'age_x', 'age_y']
iterables = [locations, durations, age_classes, age_classes]
df = pd.DataFrame(0, index=pd.MultiIndex.from_product(iterables, names=names), columns=["number_contacts", "number_survey_participants"])

# Load dataset
data = pd.read_excel(os.path.join(abs_dir, 'RawData_ComesF.xlsx'), sheet_name="CONTACT", header=[0,1,2])

# Drop columns containing NBcontacts
data.drop(columns=['NBcontact1'], inplace=True)

dropped_count=0
# Loop over all correspondents
for i in range(len(data)):
    print(f"{i/len(data)*100:.1f} % done")
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

                # if np.where(loc)[0][0] == 0:
                #     location = 'home'
                # elif np.where(loc)[0][0] == 1:
                #     location = 'school'
                # elif np.where(loc)[0][0] == 2:
                #     location = 'work_indoor'
                # elif np.where(loc)[0][0] == 3:
                #     location = 'leisure_private'
                # elif np.where(loc)[0][0] == 4:
                #     location = 'leisure_public'   
                # elif np.where(loc)[0][0] == 5:
                #     location = 'transport'
                # elif np.where(loc)[0][0] == 6:
                #     location = 'work_leisure_outdoor'

                location = locations[np.where(loc)[0][0]]


            # Assign data
            df.loc[(location, duration, age_x, age_y), 'number_contacts'] += 1
            # Remember
            participant_count.append((location, duration, age_x, age_y),)

    for index in drop_duplicates(participant_count):
        df.loc[index, 'number_survey_participants'] += 1

df['average_contacts'] = df['number_contacts']/df['number_survey_participants']

print(df.loc[('home',10,slice(None),slice(None))])
print(df.loc[('school',10,slice(None),slice(None))])
print(df.loc[('leisure_private',10,slice(None),slice(None))])
print(df.loc[('leisure_public',10,slice(None),slice(None))])
print(df.loc[('work_indoor',10,slice(None),slice(None))])
print(df.loc[('work_leisure_outdoor',10,slice(None),slice(None))])

print(f"\n{100*dropped_count/(len(data)*69):.1f} % of the {len(data)*69} reported contacts were dropped because the age, duration or location was missing\n")
