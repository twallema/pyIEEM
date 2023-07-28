import numpy as np
import pandas as pd
from functools import lru_cache
from datetime import datetime, timedelta
from pyIEEM.models.utils import ramp_fun, is_Belgian_school_holiday, aggregate_simplify_contacts

#####################
## Social contacts ##
#####################

class make_social_contact_function():

    def __init__(self, age_classes, demography, contact_type, contact_df, lmc_df, f_workplace, lav, distinguish_day_type, f_employees, conversion_matrix, simulation_start, l):
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
        
        simulation_start: datetime.datetime
            Simulation startdate. Note that this implies you can't change the simulation startdate without re-initializing the model.
            Sadly, there is no way around this (that I can think of for now).
        
        l: int
            Memory length. Should be quite large (2-3 months)
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
        self.l = l

        # pre-allocate simulation start
        if not isinstance(simulation_start, (str, datetime)):
            raise TypeError("`simulation_start` should be of type 'datetime' or 'str'")
        if isinstance(simulation_start, str):
            try:
                simulation_start = datetime.strptime(simulation_start,  "%Y-%m-%d")
            except:
                raise ValueError("conversion of `simulation_start` failed. make sure its formatted as '%Y-%m-%d'")
        self.t = simulation_start
        self.simulation_start = simulation_start

    @lru_cache()
    def __call__(self, t, M, social_restrictions, economic_closures):

        # check daytype
        vacation = is_Belgian_school_holiday(t)
        if self.distinguish_day_type:
            if ((t.weekday() == 5) | (t.weekday() == 6)):
                type_day = 'weekendday'
            else:
                type_day = 'weekday'
        else:
            type_day = 'average'

        # slice right matrices and convert to right size
        N_home, N_leisure_private, N_leisure_public, N_school, N_work = self.slice_matrices(type_day, vacation)

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

        return {'other': (1-M)*(N_home + f_school*N_school + (1-social_restrictions)*N_leisure_private + f_leisure_public*N_leisure_public), 'work': (1-M)*N_work}

    def get_contacts(self, t, states, param, tau, social_restrictions, economic_closures):
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

        ############
        ## memory ##
        ############

        # if timestep is less then 0.5 day from start the user wants a simulation restart
        start_tol = 0.5
        if abs((t - self.simulation_start)/timedelta(days=1)) < start_tol:
            # re-initialize a clean memory
            self.memory_index = [0,] #list(range(-(int(abs(self.l))), 1))
            self.memory_values = [0,] #list(np.zeros(len(self.memory_index)))
            self.I_star = 0
        # Get total number of hospitalisations (sum over all ages and spatial units)
        I = np.sum(np.sum(states['Ih'], axis=0))
        # determine length of timestep
        delta_t = (t - self.t)/timedelta(days=1)
        self.t = t
        # add case count to memory (RK23 can step backwards)
        if delta_t > 0:
            # copy values
            new_index = self.memory_index
            new_values = self.memory_values
            # append new values
            new_index.append(new_index[-1] + delta_t)
            new_values.append(I)
            # subtract the new timestep
            new_index = np.array(new_index) - new_index[-1]
            # cut of values and index to not exceed memory length l
            new_values = np.array(new_values)[new_index >= -self.l]
            new_index = new_index[new_index >= -self.l]
            # compute exponential weights at new_index
            weights = np.exp((1/tau)*new_index)/sum(np.exp((1/tau)*new_index))
            # multiply weights with case count and sum to compute average
            I_star = sum(new_values*weights)
            # update memory
            self.memory_index = list(new_index)
            self.memory_values = list(new_values)
            # update I_star
            self.I_star = I_star

        ######################
        ## behavioral model ##
        ######################

        # compute fraction of maximum IC capacity
        IC_threshold = 1000/11.6e6/(1-0.838)
        I_star = (self.I_star/np.sum(np.sum(states['S'] + states['R'], axis=0)))/IC_threshold

        # simplest model: linear relationship capped at maximum IC capacity
        M = min(I_star, 1)

        ##############
        ## policies ##
        ##############

        t_start_lockdown = datetime(2020, 3, 15) # start of lockdown
        t_end_lockdown = datetime(2020, 5, 15)

        if t < t_start_lockdown:
            return self.__call__(t, M, 0, tuple(np.zeros(63, dtype=float)))
        elif t_start_lockdown < t < t_end_lockdown:
            l = 7
            N_old = self.__call__(t, M, 0, tuple(np.zeros(63, dtype=float)))
            N_new = self.__call__(t, M, social_restrictions, tuple(economic_closures))
            return {'other': ramp_fun(t, t_start_lockdown, l, N_old['other'], N_new['other']), 'work': ramp_fun(t, t_start_lockdown, l, N_old['work'], N_new['work'])}
        else:
            l = 62
            N_old = self.__call__(t, M, social_restrictions, tuple(economic_closures))
            economic_policy = np.zeros(63, dtype=float)
            economic_policy[54] = 1
            N_new = self.__call__(t, M, 0, tuple(economic_policy))
            return {'other': ramp_fun(t, t_end_lockdown, l, N_old['other'], N_new['other']), 'work': ramp_fun(t, t_end_lockdown, l, N_old['work'], N_new['work'])}

    def slice_matrices(self, type_day, vacation):
        """
        Extract matrices on `type_day` and `vacation` and convert to np.ndarray of appropriate size
        """

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
        
        return N_home, N_leisure_private, N_leisure_public, N_school, N_work

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