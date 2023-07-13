import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

##############
## Settings ##
##############

fam = sm.families.NegativeBinomial()
ind = sm.cov_struct.Independence()

#######################################
## Locations not dependent on sector ##
#######################################

# Load data
data_df = pd.read_csv('comesf_contacts.csv', dtype={'class_size': str})

# Make sure columns are in following order
data_df = data_df.set_index(['location', 'duration', 'type_day', 'vacation'])

# pre-made dataframe containing all entries
names = ['location', 'sector', 'duration', 'type_day', 'vacation', 'age_x', 'age_y']
age_classes = data_df.reset_index()['age_x'].unique()
location = data_df.reset_index()['location'].unique()
sector = data_df.reset_index()['sector'].unique()
duration = data_df.reset_index()['duration'].unique()
type_day = data_df.reset_index()['type_day'].unique()
vacation = data_df.reset_index()['vacation'].unique()
iterables = [location, sector, duration, type_day, vacation, age_classes, age_classes]
data_GEE = pd.Series(index=pd.MultiIndex.from_product(iterables, names=names), name='desired_format', dtype=float)
data_GEE = data_GEE.sort_index()

# define relevant pure and mixed effects
pure_effects = 'reported_contacts ~ type_day + vacation + duration + age_x + age_y + sex +'
mixed_effects = 'age_x*age_y + type_day*vacation + duration*type_day + duration*vacation + duration*age_x +'
# regression formulas
formula = {
    'home': pure_effects + mixed_effects + 'household_size + professional_situation + professional_situation*duration + household_size*duration',
    'school': pure_effects + obligated_mixed_effects + '+ class_size',
    'leisure_public': pure_effects + obligated_mixed_effects + 'professional_situation',
    'leisure_private': pure_effects + obligated_mixed_effects + 'professional_situation',
    'transport': pure_effects + obligated_mixed_effects + 'professional_situation',
    'work_leisure_outdoor': pure_effects + obligated_mixed_effects + 'professional_situation',
}

print('\n')
for location, formula in formula.items():
    print(f"performing GEE in location '{location}'")
    
    # slice data
    data = data_df.loc[location].reset_index()

    # GEE regression
    mod = smf.gee(formula, "ID", data, cov_struct=ind, family=fam)
    res = mod.fit()
    
    # model fit
    data['predicted_contacts'] = res.predict()
    data = data.drop(columns=['ID', 'sex', 'household_size', 'highest_education', 'professional_situation', 'class_size', 'reported_contacts', 'sector'])
    data = data.groupby(by=['duration','type_day', 'vacation', 'age_x', 'age_y']).last()

    # fill in same value for every sector
    for sect in sector:
        data_GEE.loc[location, sect] = data['predicted_contacts'].combine_first(data_GEE.reset_index().drop(columns='sector').groupby(by=['location','duration','type_day', 'vacation', 'age_x', 'age_y']).last().loc[location].squeeze()).values
     
###################################
## Locations depending on sector ##
###################################

# define relevant pure and mixed effects
pure_effects = 'reported_contacts ~ type_day + vacation + duration + age_x + age_y + sex + professional_situation +'
mixed_effects = 'age_x*age_y + type_day*vacation + duration*type_day + duration*vacation + duration*age_x + '
formula = {
    'work_indoor': pure_effects + mixed_effects + 'sector + duration*sector + type_day*sector + vacation*sector',
    'SPC': pure_effects + mixed_effects + 'sector + duration*sector',
}
for location, formula in formula.items():
    print(f"performing GEE in location '{location}'")
    
    # slice data
    data = data_df.loc[location].reset_index()

    # GEE regression
    mod = smf.gee(formula, "ID", data, cov_struct=ind, family=fam)
    res = mod.fit()

    # model fit
    data['predicted_contacts'] = res.predict()
    data = data.drop(columns=['ID', 'sex', 'household_size', 'highest_education', 'professional_situation', 'class_size', 'reported_contacts'])
    data = data.groupby(by=['sector', 'duration','type_day', 'vacation', 'age_x', 'age_y']).last()

    # data post GEE
    data_GEE.loc[location] = data['predicted_contacts'].combine_first(data_GEE.loc[location]).values

# write to .csv
data_GEE.to_csv('comesf_matrices.csv')

###################
## Visualisation ##
###################

# # visualize result
# location = 'work_indoor'
# sector = 'C'
# vacation = False
# # slice data
# sl = data_GEE.groupby(by=['location', 'sector', 'type_day', 'vacation', 'age_x', 'age_y']).sum()

# # compute average week/weekend
# week = sl.loc[location, sector, 'weekday', vacation].values.reshape(2*[len(age_classes),])
# weekend = sl.loc[location, sector, 'weekendday', vacation].values.reshape(2*[len(age_classes),])
# m = week

# # visualise result
# fig,ax=plt.subplots()
# ax = sns.heatmap(m, annot=True, fmt='.2f', ax=ax, square=True, cbar=False, annot_kws={"size":6})
# ax.xaxis.tick_top() # x axis on top
# ax.xaxis.set_label_position('top')
# ax.set_xticklabels(age_classes, rotation = 30, size=8)
# ax.set_yticklabels(age_classes, rotation = 30, size=8)
# ax.set_title(f'at {location}, weekend/week average ({vacation})\navg. contacts: {np.mean(np.sum(m,axis=1)):.2f}', fontsize=10)
# ax.set(ylabel='participant age')
# plt.show()
# plt.close()