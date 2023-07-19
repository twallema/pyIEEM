""" A script to format the raw dataset by CoMEs-F in a more pleasant and easy-to-use format
"""

import os
import sys
import math
import numpy as np
import pandas as pd 
from tqdm import tqdm
from pyIEEM.data.utils import convert_age_stratified_quantity

# Define desired age groups of final matrices
age_classes = pd.IntervalIndex.from_tuples([(0,5),(5,10),(10,15),(15,20),(20,25),(25,30),(30,35),(35,40),(40,45),(45,50),(50,55),(55,60),(60,65),(65,70),(70,75),(75,80),(80,120)], closed='left')

# Helper function
def drop_duplicates(input_list):
    unique_tuples = {}
    result = []
    duplicates_count = []
    for entry in input_list:
        key = (entry[0], entry[1], frozenset(entry[2]), entry[3], entry[4], entry[5], entry[6],
                entry[7], frozenset(entry[8]), entry[9], entry[10], entry[11], entry[12])
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

# Load demography of France
demo_df = pd.read_csv(os.path.join(abs_dir, '../../../../../../data/interim/epi/demographic/age_structure_FR_2019.csv'), index_col=[0,1]).squeeze().droplevel(0)

# Load dataset
data = pd.read_excel(os.path.join(abs_dir, 'RawData_ComesF.xlsx'), sheet_name="CONTACT", header=[0,1,2])
data.sort_index(axis=1).drop(columns=['NBcontact1'], inplace=True)

# Translation
translations = {
    'class_sizes': ['< 20', '20 - 30', '> 30'],
    'highest_educations': ['primary', 'primary', 'secundary', 'secundary', 'tertiary', 'tertiary'],
    'professional_situations': 6*['employed']+['student', 'retired', 'unemployed', 'unemployed'],
    'sexes': ['F', 'M'],
    'sector': ['A', 'C10-12', 'C', 'D', 'F', 'G', 'K', 'M', 'S, T', 'P, Q', 'O, N'],
    'location': ['home', 'school', 'work_indoor', 'leisure_private', 'leisure_public', 'transport', 'work_leisure_outdoor', 'SPC'],
    'duration': ['< 5 min', '5-15 min', '15-60 min', '60-240 min', '> 240 min'],
    'age_y': [str(a) for a in age_classes.values],
    'daytype': ['weekendday', 'weekday'],
    'vacation': [True, False],
    'age_group_SPC': pd.IntervalIndex.from_tuples([(0,5),(5,10),(10,18),(18,65),(65,120)], closed='left'),
}

# Pre-allocate dictionary for the output
columns = ['ID', 'sex', 'age_x', 'household_size', 'class_size', 'highest_education', 'professional_situation',
            'sector', 'age_y', 'location', 'duration', 'daytype', 'vacation', 'reported_contacts']
output = {k: [] for k in columns}

# Pre-allocate dictionary for the SPC (no location, duration, daytype or vacation available)
columns = ['ID', 'sex', 'age_x', 'household_size', 'class_size', 'highest_education', 'professional_situation', 'sector', 'age_y', 'reported_contacts']
output_SPC = {k: [] for k in columns}

## PARTICIPANT LOOP
dropped_count_personal=dropped_count_temporal=dropped_count_contact=0
for i in tqdm(range(len(data))):

    row = data.iloc[i]

    # Assign ID to survey participant
    ID = i

    ## Personal characteristics
    # missing age, sex, household size or professional situation
    if ((math.isnan(row['Q1']['Q3_1']["Age du sujet de l'enquête"])) | \
        (math.isnan(row['Q2']['Q2']["Sexe du sujet de l'enquête"])) | \
        (math.isnan(row['Q7nb']['Q3nb']["Nombre de personnes au foyer"])) | \
        (math.isnan(row['Q9']['Q6']['situation professionnelle ']))):

        dropped_count_personal += 1
        continue

    # age
    age = row['Q1']['Q3_1']["Age du sujet de l'enquête"]
    age_x = age_classes[age_classes.contains(age)].values
    age_x = age_x.astype(str)[0]
    # sex
    sex = translations['sexes'][int(row['Q2']['Q2']["Sexe du sujet de l'enquête"]-1)]
    # household size
    household_size = int(row['Q7nb']['Q3nb']["Nombre de personnes au foyer"])
    # professional situation
    professional_situation = translations['professional_situations'][int(row['Q9']['Q6']['situation professionnelle ']-1)]
    if professional_situation == 'student':
        if ((not math.isnan(row['SPC']['Q9']["Nombre d'étudiants dans la classe"])) & \
            (row['SPC']['Q9']["Nombre d'étudiants dans la classe"] <=3) & \
            (row['SPC']['Q9']["Nombre d'étudiants dans la classe"] >= 1)):
            class_size = translations['class_sizes'][int(row['SPC']['Q9']["Nombre d'étudiants dans la classe"])-1]
        else:
            class_size = 'not_applicable'
            dropped_count_personal += 1
            continue
    else:
        class_size = 'not_applicable'
    # sector of employment
    sector_n = row['Q10']['Q7']["Dans quel secteur d'activité travaillez-vous"]
    if ((not math.isnan(sector_n)) & (sector_n >= 1) & (sector_n <= 11)):
        sector = translations['sector'][int(sector_n - 1)]
        # aggregate the obserations of C10-12 into C
        if sector == 'C10-12':
            sector = 'C'
    else:
        sector = 'not_applicable'
    # highest diploma
    if not math.isnan(row['Q8']['Q4']['diplôme le plus élevé']):
        highest_education = translations['highest_educations'][int(row['Q8']['Q4']['diplôme le plus élevé']-1)]
    else:
        if row['Q1']['Q3_1']["Age du sujet de l'enquête"] <= 15:
            highest_education = 'primary'
        else:
            dropped_count_personal+=1
            continue
    # parent filled in survey --> professional situation/economic sector filled in by parent --> correct this, below 16 everyone is a student
    if ((age <= 16) & (professional_situation != 'student')):
        professional_situation = 'student'
        sector = 'not_applicable'
        highest_education = 'primary'
    # supplementary professional contacts
    SPC_data = row['SPC'].droplevel([0]).values
    contact_properties_SPC = []
    contact_count_SPC = []
    if ((not math.isnan(SPC_data[0])) & (SPC_data[0] == 1) & (sector != 'not_applicable')):
        if ((not math.isnan(SPC_data[1])) & (sum(SPC_data[2:7]) != 0)):
            if not age <= 16:
                # Distribute the total number of contacts over the age groups indicated by the survey participant using demographic weighing
                n = SPC_data[1]
                # Cap at 134 contacts --> reported 95% quantile by Beraud
                if n > 200:
                    n = 200
                age_groups_SPC = translations['age_group_SPC'][SPC_data[2:7] != 0]
                d = pd.Series(n, index=pd.IntervalIndex.from_tuples([(0,105),], closed='left'))
                d = convert_age_stratified_quantity(d, age_groups_SPC, demo_df)
                d = d/sum(d)*n
                # Convert to the desired age groups of the contact matrices
                out = convert_age_stratified_quantity(d, age_classes, demo_df)
                # Add the contact properties (these are always "unique" so we can paste them after ommitting the unique contacts down below)
                for age_y in out.index:
                    # Keep track of the contacts data so we can eliminate doubles in the end
                    contact_properties_SPC.append((ID, sex, age_x, household_size, class_size, highest_education, professional_situation, sector, str(age_y),),)
                    contact_count_SPC.append(out[age_y])

    ## Temporal characteristics

    # Temporal characteristics: type of day
    days_of_week = row.loc['Jour1'].droplevel(0).values
    # Temporal characteristics: vacation or not
    is_vacation = row.loc['VAC1'].droplevel(0).values
    # Perform checks on impossible day types
    if ((any(math.isnan(d) for d in days_of_week)) | (any(d > 7 for d in days_of_week)) | (any(d < 1 for d in days_of_week))):
        dropped_count_temporal+=1
        continue

    # CONTACT LOOP
    else:
        # Keep track of all contact's properties
        contact_properties=[]
        for j in range(1,69):
            # Extract correct daytype/vacation
            if j <= 40:
                vacation = is_vacation[0]
                if vacation == 1:
                    vacation = True
                elif vacation == 2:
                    vacation = False
                daytype = days_of_week[0]
                if daytype <= 5:
                    daytype = 'weekday'
                else:
                    daytype = 'weekendday'
            else:
                vacation = is_vacation[1]
                if vacation == 1:
                    vacation = True
                elif vacation == 2:
                    vacation = False
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
                    dropped_count_contact+=1
                else:
                    # Extract age of contacted person and duration
                    age_y = age_classes[age_classes.contains(contact_data.loc['age moyen '])].values
                    age_y = age_y.astype(str)[0]
                    duration = translations['duration'][int(contact_data.loc['durée ']-1)]
                    # Extract location, keep track of dropped contacts due to missing location
                    loc = contact_data.iloc[4:11].values
                    loc = loc!= 0               
                    location = translations['location'][np.where(loc)[0][0]]
                    # Some parents have filed teacher - student contacts of their children as work_indoor --> set to school below age 16
                    if ((age <= 16) & (location == 'work_indoor')):
                        location = 'school'
                        
                # Keep track of the contacts data so we can eliminate doubles
                contact_properties.append((ID, sex, age_x, household_size, class_size, highest_education, professional_situation,
                                            sector, age_y, location, duration, daytype, vacation),)

        # Drop the duplicate data
        unique_indices, contact_counts = drop_duplicates(contact_properties)

        # Append to output dictionary
        output['reported_contacts'].extend(contact_counts)
        for unique_index in unique_indices:
            for i, key in enumerate(output.keys()):
                if key != 'reported_contacts':
                    output[key].append(unique_index[i])

        # Append to SPC output dictionary
        output_SPC['reported_contacts'].extend(contact_count_SPC)
        for unique_index in contact_properties_SPC:
            for i, key in enumerate(output_SPC.keys()):
                if key != 'reported_contacts':
                    output_SPC[key].append(unique_index[i])

# Print the fraction of dropped contacts
print(f"\n{100*dropped_count_personal/(len(data)*69):.1f} % of the {len(data)*69} reported contacts were dropped because the age, sex, household size or highest education level were missing")
print(f"{100*dropped_count_temporal/(len(data)*69):.1f} % of the {len(data)*69} reported contacts were dropped because the date listed on the survey was invalid")
print(f"{100*dropped_count_contact/(len(data)*69):.1f} % of the {len(data)*69} reported contacts were dropped because the age, duration, or location of the contact was missing\n")

# Define output dataframe containing all contact-related characteristics in index and person-related characteristics in the columns
names = ['ID', 'age_y', 'location', 'duration', 'type_day', 'vacation']
iterables = [np.unique(output['ID']), translations['age_y'], translations['location'], translations['duration'], translations['daytype'], translations['vacation']]
df = pd.DataFrame(0, index=pd.MultiIndex.from_product(iterables, names=names), columns=['sex', 'age_x', 'household_size', 'class_size', 'highest_education', 'professional_situation', 'sector', 'reported_contacts'])

# Fill in correct personal characteristics for every ID
for id in tqdm(np.unique(output['ID'])):
    for personal_characterstic in ['sex', 'age_x', 'household_size', 'class_size', 'highest_education', 'professional_situation', 'sector']:
        df.loc[df.index.get_level_values('ID') == id, personal_characterstic] = str(np.unique(np.array(output[personal_characterstic])[output['ID']==id])[0])
    
# Fill in present values
for ID, age_x, sector, age_y, location, duration, daytype, vacation, contacts in zip(output['ID'], output['age_x'], output['sector'], output['age_y'], output['location'], output['duration'], output['daytype'], output['vacation'], output['reported_contacts']):
    df.loc[(ID, age_y, location, duration, daytype, vacation),'reported_contacts'] = contacts

# rearrange columns: 
column_order = ['location','duration','type_day','vacation', 'ID', 'age_x', 'sex', 'household_size', 'class_size', 'highest_education', 'professional_situation', 'sector', 'age_y', 'reported_contacts']
df = df.reset_index()
df = df[column_order]

## compute the distribution of durations of work contacts per sector, for individuals who reported between 10 and 20 contacts
# output dataframe
iterables = [df['sector'].unique(), df['duration'].unique()]
dur = df['duration'].unique()
names = ['sector', 'duration']
duration_distribution = pd.Series(0,index=pd.MultiIndex.from_product(iterables, names=names), name='duration_distribution', dtype=float)
# groupby for convenience
df = df.groupby(by=['ID', 'location', 'duration', 'type_day', 'vacation', 'age_y']).last()
# loop over all participants
for ID in tqdm(df.index.get_level_values('ID').unique()):
    # slice location work_indoor on a weekday outside vacation
    sl = df.loc[ID, 'work_indoor', slice(None), 'weekday', False, slice(None)]
    # particpant has more than 10 work contacts?
    if not sum(sl['reported_contacts'].groupby(by=['duration']).sum()) > 8:
        continue
    else:
        # sector
        sector = sl['sector'].unique()
        # distribution
        duration_distribution.loc[sector, slice(None)] += sl['reported_contacts'].groupby(by='duration').sum().reindex(index = dur).values
# global average distribution
glob = duration_distribution.groupby(by='duration').sum()/sum(duration_distribution)
# compute average distribution per sector --> no data --> global average assumed
for sector in duration_distribution.index.get_level_values('sector').unique():
    n = sum(duration_distribution.loc[sector])
    if not n ==0:
        duration_distribution.loc[sector] = (duration_distribution.loc[sector]/sum(duration_distribution.loc[sector])).values
    else:
        duration_distribution.loc[sector] = glob.values

## Split every SPC in accordance with the duration distribution
n = len(duration_distribution.index.get_level_values('duration').unique())
# personal properties
ID = []
sex = []
age_x = []
household_size = []
class_size = []
highest_education = []
professional_situation = []
sector = []
age_y = []
location = []
duration = []
daytype = []
vacation = []
# contacts
reported_contacts = []
duration = []
# Loop over SPC output
for i in tqdm(range(len(output_SPC['ID']))):
    # new in the output
    location.extend(n*['SPC',])
    daytype.extend(n*['weekday',])
    vacation.extend(n*[False,])
    # blow personal properties up
    ID.extend(n*[output_SPC['ID'][i],])
    sex.extend(n*[output_SPC['sex'][i],])
    age_x.extend(n*[output_SPC['age_x'][i],])
    household_size.extend(n*[output_SPC['household_size'][i],])
    class_size.extend(n*[output_SPC['class_size'][i],])
    highest_education.extend(n*[output_SPC['highest_education'][i],])
    professional_situation.extend(n*[output_SPC['professional_situation'][i],])
    sector.extend(n*[output_SPC['sector'][i],])
    age_y.extend(n*[output_SPC['age_y'][i],])
    # distribute contacts over durations
    reported_contacts.extend(output_SPC['reported_contacts'][i]*duration_distribution.loc[output_SPC['sector'][i]].values)
    # make a list of durations
    duration.extend(duration_distribution.index.get_level_values('duration').unique(),)

# Update SPC output dictionary
output_SPC.update({
    'ID': ID,
    'sex': sex,
    'age_x': age_x,
    'household_size': household_size,
    'class_size': class_size,
    'highest_education': highest_education,
    'professional_situation': professional_situation,
    'sector': sector,
    'age_y': age_y,
    'location': location,
    'duration': duration,
    'daytype': daytype,
    'vacation': vacation,
    'reported_contacts': reported_contacts
})

# merge SPC output and regular output
for k,v in output_SPC.items():
    output[k].extend(v)

# construct dataframe
df_small = pd.DataFrame(output).set_index('ID').sort_index()

# Define output dataframe containing all contact-related characteristics in index and person-related characteristics in the columns
names = ['ID', 'age_y', 'location', 'duration', 'daytype', 'vacation']
iterables = [np.unique(output['ID']), translations['age_y'], translations['location'], translations['duration'], translations['daytype'], translations['vacation']]
df = pd.DataFrame(0, index=pd.MultiIndex.from_product(iterables, names=names), columns=['sex', 'age_x', 'household_size', 'class_size', 'highest_education', 'professional_situation', 'sector', 'reported_contacts'])

# Fill in correct personal characteristics for every ID
for id in tqdm(np.unique(output['ID'])):
    for personal_characterstic in ['sex', 'age_x', 'household_size', 'class_size', 'highest_education', 'professional_situation', 'sector']:
        df.loc[df.index.get_level_values('ID') == id, personal_characterstic] = str(np.unique(np.array(output[personal_characterstic])[output['ID']==id])[0])

# Fill in present values
for ID, age_x, sector, age_y, location, duration, daytype, vacation, contacts in zip(output['ID'], output['age_x'], output['sector'], output['age_y'], output['location'], output['duration'], output['daytype'], output['vacation'], output['reported_contacts']):
    df.loc[(ID, age_y, location, duration, daytype, vacation),'reported_contacts'] = contacts

# Remove daytypes and vacations not originally present (this would mean we're adding extra days to the survey)
df_no_index = df.reset_index()
for id in tqdm(np.unique(output['ID'])):
    dt_list = np.unique(np.array(output['daytype'])[output['ID']==id])
    vctns_list = np.unique(np.array(output['vacation'])[output['ID']==id])

    dt_list_complement = [d for d in translations['daytype'] if d not in dt_list]
    vctns_list_complement = [d for d in translations['vacation'] if d not in vctns_list]
    for dt in dt_list_complement: 
        df_no_index.drop(df_no_index[(df_no_index.ID == id) & (df_no_index.daytype == dt)].index, inplace = True)
    for vctns in vctns_list_complement:
        df_no_index.drop(df_no_index[(df_no_index.ID == id) & (df_no_index.vacation == vctns)].index, inplace = True)
df = df_no_index
df = df.set_index('ID').sort_index()

# rearrange columns: 
column_order = ['location','duration','daytype', 'vacation', 'age_x', 'sex', 'household_size', 'class_size', 'highest_education', 'professional_situation', 'sector', 'age_y', 'reported_contacts']
df = df[column_order]
df = df.rename(columns={'daytype': 'type_day'})

# Save normal contacts
df.to_csv('comesf_formatted_data.csv')
