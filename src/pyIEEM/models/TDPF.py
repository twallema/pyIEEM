import numpy as np
import pandas as pd
from functools import lru_cache
from datetime import datetime, timedelta
from pyIEEM.data.utils import aggregate_contact_matrix

#####################
## Social contacts ##
#####################

class make_social_contact_function():

    def __init__(self, age_classes, demography, contact_type, contact_df, lmc_df, f_workplace, lav, distinguish_day_type, f_employees, conversion_matrix):
        """
        Time-dependent parameter function of social contacts

        input
        =====

        age_classes: pd.IntervalIndex
            age classes of the epidemiological model

        demography: pd.Series
            demography of country under study

        contact_type: str
            'absolute_contacts' versus 'integrated_contacts'

        contact_df: pd.DataFrame
            number of social contacts per location, sector, daytype and vacation.
        
        lmc_df: pd.Series
            Labor market composition (fraction employed in economic activity of NACE 21) per spatial patch in the model. 
            Index: ['spatial_unit', 'sector']

        f_workplace: pd.Series
            Fraction of employees at workplace per sector of NACE 64 during first 2020 Belgian COVID-19 lockdown.
            Index: NACE64 sectors

        lav: pd.Series
            Leisure association vector. Contains one if sector of NACE 64 is associated with public leisure contacts. Contains zero otherwise.
            Index: NACE 64 sectors

        distinguish_day_type: bool
            Return weekendday/weekday average number of contacts or return weekday/weekendday contacts depending on simulation day of week.
        
        f_employees: pd.Series
            Fraction of employees in a given NACE 21 sector, working in a NACE 64 sector

        conversion_matrix: np.ndarray
            NACE 64 to NACE 21 conversion matrix.
        """

        # input checks
        assert isinstance(age_classes, pd.IntervalIndex)
        assert isinstance(demography, pd.Series)
        assert ((contact_type == 'absolute_contacts') | (contact_type == 'integrated_contacts'))

        # infer the number of spatial patches
        if 'spatial_unit' in lmc_df.index.names:
            self.G = len(lmc_df.index.get_level_values('spatial_unit').unique().values)
        else:
            self.G = 1

        # Extract contact matrices and demographically convert to right age bins
        self.N_home, self.N_leisure_private, self.N_leisure_public, self.N_school, self.N_work = aggregate_simplify_contacts(contact_df, age_classes, demography, contact_type)
        # Extract contact matrices of contacts < 5 min and demographically convert to right age bins
        self.N_home_short, self.N_leisure_private_short, self.N_leisure_public_short, self.N_school_short, self.N_work_short = aggregate_simplify_contacts(contact_df, age_classes, demography, contact_type+'_less_5_min')

        # Assign to variables
        self.age_classes = age_classes
        self.distinguish_day_type = distinguish_day_type
        self.lmc_df = lmc_df
        self.f_workplace = f_workplace
        self.lav = lav
        self.f_employees = f_employees
        self.conversion_matrix = conversion_matrix
    
    @lru_cache()
    def __call__(self, t, social_restrictions, preventive_measures, economic_closures):

        ## check daytype
        vacation = is_Belgian_primary_secundary_school_holiday(t)
        if self.distinguish_day_type:
            if ((t.weekday() == 5) | (t.weekday() == 6)):
                type_day = 'weekendday'
            else:
                type_day = 'weekday'
        else:
            type_day = 'average'

        ## slice right matrices and convert to right size

        # long contacts
        # Home, leisure, school: (age, age, spatial_unit)
        if self.G != 1:
            N_home = np.tile(np.expand_dims(self.N_home.loc[type_day, vacation, slice(None), slice(None)].values.reshape(2*[len(self.age_classes),]), axis=2), self.G)
            N_leisure_private = np.tile(np.expand_dims(self.N_leisure_private.loc[type_day, vacation, slice(None), slice(None)].values.reshape(2*[len(self.age_classes),]), axis=2), self.G)
            N_leisure_public = np.tile(np.expand_dims(self.N_leisure_public.loc[type_day, vacation, slice(None), slice(None)].values.reshape(2*[len(self.age_classes),]), axis=2), self.G)
            N_school = np.tile(np.expand_dims(self.N_school.loc[type_day, vacation, slice(None), slice(None)].values.reshape(2*[len(self.age_classes),]), axis=2), self.G)
        else:
            N_home = np.expand_dims(self.N_home.loc[type_day, vacation, slice(None), slice(None)].values.reshape(2*[len(self.age_classes),]), axis=2)
            N_leisure_private = np.expand_dims(self.N_leisure_private.loc[type_day, vacation, slice(None), slice(None)].values.reshape(2*[len(self.age_classes),]), axis=2)
            N_leisure_public = np.expand_dims(self.N_leisure_public.loc[type_day, vacation, slice(None), slice(None)].values.reshape(2*[len(self.age_classes),]), axis=2)
            N_school = np.expand_dims(self.N_school.loc[type_day, vacation, slice(None), slice(None)].values.reshape(2*[len(self.age_classes),]), axis=2)
        # Work: (NACE 21, age, age)
        N_work = np.swapaxes(self.N_work.loc[slice(None), type_day, vacation, slice(None), slice(None)].values.reshape([len(self.N_work.index.get_level_values('sector').unique().values),] +2*[len(self.age_classes),]), 0, -1)

        # short contacts
        # Home, leisure, school: (age, age, spatial_unit)
        if self.G != 1:
            N_home_short = np.tile(np.expand_dims(self.N_home_short.loc[type_day, vacation, slice(None), slice(None)].values.reshape(2*[len(self.age_classes),]), axis=2), self.G)
            N_leisure_private_short = np.tile(np.expand_dims(self.N_leisure_private_short.loc[type_day, vacation, slice(None), slice(None)].values.reshape(2*[len(self.age_classes),]), axis=2), self.G)
            N_leisure_public_short = np.tile(np.expand_dims(self.N_leisure_public_short.loc[type_day, vacation, slice(None), slice(None)].values.reshape(2*[len(self.age_classes),]), axis=2), self.G)
            N_school_short = np.tile(np.expand_dims(self.N_school_short.loc[type_day, vacation, slice(None), slice(None)].values.reshape(2*[len(self.age_classes),]), axis=2), self.G)
        else:
            N_home_short = np.expand_dims(self.N_home_short.loc[type_day, vacation, slice(None), slice(None)].values.reshape(2*[len(self.age_classes),]), axis=2)
            N_leisure_private_short = np.expand_dims(self.N_leisure_private_short.loc[type_day, vacation, slice(None), slice(None)].values.reshape(2*[len(self.age_classes),]), axis=2)
            N_leisure_public_short = np.expand_dims(self.N_leisure_public_short.loc[type_day, vacation, slice(None), slice(None)].values.reshape(2*[len(self.age_classes),]), axis=2)
            N_school_short = np.expand_dims(self.N_school_short.loc[type_day, vacation, slice(None), slice(None)].values.reshape(2*[len(self.age_classes),]), axis=2)
        # Work: (NACE 21, age, age)
        N_work_short = np.swapaxes(self.N_work_short.loc[slice(None), type_day, vacation, slice(None), slice(None)].values.reshape([len(self.N_work_short.index.get_level_values('sector').unique().values),] +2*[len(self.age_classes),]), 0, -1)

        # subtract short contacts from long contacts
        N_home -= preventive_measures*N_home_short
        N_leisure_private -= preventive_measures*N_leisure_private_short
        N_leisure_public -= preventive_measures*N_leisure_public_short
        N_school -= preventive_measures*N_school_short
        N_work -= preventive_measures*N_work_short

        # convert economic policy from tuple to numpy array
        assert isinstance(economic_closures, tuple)
        economic_closures = 1-np.array(economic_closures, dtype=float)

        # assert degree of school opennness
        f_school = economic_closures[np.where(self.f_workplace.index == 'P85')[0][0]]

        # convert the leisure_public contacts depending on the economic policy
        f_leisure_public = sum(self.lav * economic_closures)

        # zero in economic_closures corresponds to full lockdown in Belgium
        economic_closures = self.f_workplace.values + economic_closures*(1-self.f_workplace.values)

        # convert economic policy from NACE 64 to NACE 21
        economic_closures = np.matmul(economic_closures*self.f_employees.values, self.conversion_matrix)

        # multiply the work contacts (age, age, NACE 21) with the openness of the sectors
        N_work = N_work*economic_closures[np.newaxis, np.newaxis, :]

        # convert work contacts to (age, age, spatial_unit) using the labor market structure
        N_work = np.einsum('ijk, kl -> ijl', N_work, np.transpose(self.lmc_df.values.reshape([self.G, len(economic_closures)])))

        return {'other': N_home + 0.4*(f_school*N_school + (1-social_restrictions)*N_leisure_private + f_leisure_public*N_leisure_public), 'work': 0.4*N_work}

    def get_contacts(self, t, states, param, social_restrictions, preventive_measures, economic_closures):
        """
        Function returning the number of social contacts under sector closure and/or lockdown

        input
        =====

        output
        ======

        N: dict
            Keys: "work" and "other"
            Contact matrix per spatial patch at work and in all other locations.
        """

        t_start_lockdown = datetime(2020, 3, 15) # start of lockdown
        t_end_lockdown = datetime(2020, 5, 15)

        if t < t_start_lockdown:
            return self.__call__(t, 0, 0, tuple(np.zeros(63, dtype=float)))
        elif t_start_lockdown < t < t_end_lockdown:
            l = 7
            N_old = self.__call__(t, 0, 0, tuple(np.zeros(63, dtype=float)))
            N_new = self.__call__(t, social_restrictions, preventive_measures, tuple(economic_closures))
            return {'other': ramp_fun(t, t_start_lockdown, l, N_old['other'], N_new['other']), 'work': ramp_fun(t, t_start_lockdown, l, N_old['work'], N_new['work'])}
        else:
            l = 62
            N_old = self.__call__(t, social_restrictions, preventive_measures, tuple(economic_closures))
            economic_policy = np.zeros(63, dtype=float)
            economic_policy[54] = 1
            N_new = self.__call__(t, 0, 1, tuple(economic_policy))
            return {'other': ramp_fun(t, t_end_lockdown, l, N_old['other'], N_new['other']), 'work': ramp_fun(t, t_end_lockdown, l, N_old['work'], N_new['work'])}


def ramp_fun(t, t_start, l, N_old, N_new):
    """

    input
    =====

    t : timestamp
        current date

    t_start : timestamp
        start of ramp

    l : timestamp
        length of ramp

    N_old : float/np.array
        old policy
    
    N_new : float/np.array
        new policy

    output
    ======

    N_t : float/np.array
        interpolation between N_old and N_new
    """
    if t_start < t < t_start+timedelta(days=l):
        return N_old + ((N_new-N_old)/l)*((t-t_start)/timedelta(days=1))
    else:
        return N_new

from dateutil.easter import easter
def is_Belgian_primary_secundary_school_holiday(d):
    """
    A function returning 'True' if a given date is a school holiday or primary and secundary schools in Belgium
    
    Input
    -----
    
    d: datetime.datetime
        Current simulation date
    
    Returns
    -------
    
    is_Belgian_primary_secundary_school_holiday: bool
        True: date `d` is a school holiday for primary and secundary schools
    """
    
    # Pre-allocate a vector containing the year's holiday weeks
    holiday_weeks = []
    
    # Herfstvakantie
    holiday_weeks.append(44)
    
    # Extract date of easter
    d_easter = easter(d.year)
    # Convert from datetime.date to datetime.datetime
    d_easter = datetime(d_easter.year, d_easter.month, d_easter.day)
    # Get week of easter
    w_easter = d_easter.isocalendar().week

    # Default logic: Easter holiday starts first monday of April
    # Unless: Easter falls after 04-15: Easter holiday ends with Easter
    # Unless: Easter falls in March: Easter holiday starts with Easter
    if d_easter >= datetime(year=d.year,month=4,day=15):
        w_easter_holiday = w_easter - 1
    elif d_easter.month == 3:
        w_easter_holiday = w_easter + 1
    else:
        w_easter_holiday = datetime(d.year, 4, (8 - datetime(d.year, 4, 1).weekday()) % 7).isocalendar().week
    holiday_weeks.append(w_easter_holiday)
    holiday_weeks.append(w_easter_holiday+1)

    # Krokusvakantie
    holiday_weeks.append(w_easter-6)

    # Extract week of Christmas
    # If Christmas falls on Saturday or Sunday, Christams holiday starts week after
    w_christmas_current = datetime(year=d.year,month=12,day=25).isocalendar().week
    if datetime(year=d.year,month=12,day=25).isoweekday() in [6,7]:
        w_christmas_current += 1
    w_christmas_previous = datetime(year=d.year-1,month=12,day=25).isocalendar().week
    if datetime(year=d.year-1,month=12,day=25).isoweekday() in [6,7]:
        w_christmas_previous += 1
    # Christmas logic
    if w_christmas_previous == 52:
        if datetime(year=d.year-1, month=12, day=31).isocalendar().week != 53:
            holiday_weeks.append(1)   
    if w_christmas_current == 51:
        holiday_weeks.append(w_christmas_current)
        holiday_weeks.append(w_christmas_current+1)
    if w_christmas_current == 52:
        holiday_weeks.append(w_christmas_current)
        if datetime(year=d.year, month=12, day=31).isocalendar().week == 53:
            holiday_weeks.append(w_christmas_current+1)

    # Define Belgian Public holidays
    public_holidays = [
        datetime(year=d.year, month=1, day=1),       # New Year
        d_easter + timedelta(days=1),                # Easter monday
        datetime(year=d.year, month=5, day=1),       # Labor day
        d_easter + timedelta(days=40),               # Acension Day
        datetime(year=d.year, month=7, day=21),      # National holiday
        datetime(year=d.year, month=8, day=15),      # Assumption Mary
        datetime(year=d.year, month=11, day=1),      # All Saints
        datetime(year=d.year, month=11, day=11),     # Armistice
        datetime(year=d.year, month=12, day=25),     # Christmas
    ]
    
    # Logic
    if ((d.isocalendar().week in holiday_weeks) | \
            (d in public_holidays)) | \
                ((datetime(year=d.year, month=7, day=1) <= d < datetime(year=d.year, month=9, day=1))):
        return True
    else:
        return False

def aggregate_simplify_contacts(contact_df, age_classes, demography, contact_type):
    """
    A function to demographically convert the contacts to the right age classes and simplify the (very large) pd.DataFrame `contact_df`

    input
    =====

    contact_df: pd.DataFrame
        contacts obtained from Beraud (2015)

    age_classes: pd.IntervalIndex
        age classes of the epidemiological model

    demography: pd.Series
        demography of country under study.

    contact_type: str
        'absolute_contacts'/'absolute_contacts_less_5_min' versus 'integrated_contacts'/'integrated_contacts_less_5_min'

    output
    ======

    N_home: pd.Series
        Home contacts. Index: [day_type, vacation, age_x, age_y]. Values: `contact_type`

    N_leisure_private: pd.Series
        Private leisure contacts. Idem.

    N_leisure_public: pd.Series
        Public leisure contacts. Idem.

    N_school: pd.Series
        School contacts. Idem.

    N_work: pd.Series
        Work contacts. Index: [sector, day_type, vacation, age_x, age_y]. Values: `contact_type`
    """

    # Determine if a demographic conversion is necessary (this is computationally demanding and takes approx. 1 min.)
    if all(age_classes == pd.IntervalIndex.from_tuples([(0,5),(5,10),(10,15),(15,20),(20,25),(25,30),(30,35),(35,40),(40,45),(45,50),(50,55),(55,60),(60,65),(65,70),(70,75),(75,80),(80,120)], closed='left')):
        convert = False
    else:
        convert = True

    matrices=[]
    for location in contact_df.index.get_level_values('location').unique().values:
        if location not in ['SPC', 'work']:
            # output dataframe
            type_days = contact_df.index.get_level_values('type_day').unique().values
            vacations = contact_df.index.get_level_values('vacation').unique().values
            age_x = age_y = age_classes
            iterables = [type_days, vacations, age_x, age_y]
            contact = pd.Series(index=pd.MultiIndex.from_product(iterables, names=['type_day', 'vacation', 'age_x', 'age_y']), name=location, dtype=float)                           
            for type_day in type_days:
                for vacation in vacations:
                    m = contact_df.loc[location, 'A', type_day, vacation, slice(None), slice(None)][contact_type]
                    if convert:
                        m = aggregate_contact_matrix(m, age_classes, demography)
                    contact.loc[type_day, vacation, slice(None), slice(None)] = m.values
            matrices.append(contact)

        elif location == 'work':
            # output dataframe
            sectors = contact_df.index.get_level_values('sector').unique().values
            type_days = contact_df.index.get_level_values('type_day').unique().values
            vacations = contact_df.index.get_level_values('vacation').unique().values
            age_x = age_y = age_classes
            iterables = [sectors, type_days, vacations, age_x, age_y]
            contact = pd.Series(index=pd.MultiIndex.from_product(iterables, names=['sector', 'type_day', 'vacation', 'age_x', 'age_y']), name=location, dtype=float)                           
            for sector in sectors:
                for type_day in type_days:
                    for vacation in vacations:
                        m = contact_df.loc[location, sector, type_day, vacation, slice(None), slice(None)][contact_type]
                        if convert:
                            m = aggregate_contact_matrix(m, age_classes, demography)
                        contact.loc[sector, type_day, vacation, slice(None), slice(None)] = m.values
            matrices.append(contact)

    return (*matrices,)

#################
## Seasonality ##
#################

class make_seasonality_function():
    """
    Simple class to create functions that controls the season-dependent value of the transmission coefficients.
    """
    def __call__(self, t, states, param, amplitude):
        """
        Default output function. Returns the transmission coefficient beta multiplied with a sinusoid with average value 1.
        
        t : datetime.datetime
            simulation time
        amplitude : float
            maximum deviation of output with respect to the average (1)
        """
        maxdate = datetime(2021,1,1)
        # One period is one year long (seasonality)
        t = (t - maxdate)/timedelta(days=1)/365
        rescaling = 1 + amplitude*np.cos( 2*np.pi*(t))
        return param*rescaling