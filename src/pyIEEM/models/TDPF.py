import numpy as np
import pandas as pd
from functools import lru_cache
from datetime import datetime, timedelta
from pyIEEM.models.utils import ramp_fun, is_school_holiday, aggregate_simplify_contacts

#####################
## Social contacts ##
#####################

class make_social_contact_function():

    def __init__(self, age_classes, demography, contact_type, contact_df, lmc_df, f_workplace, f_remote, hesitancy, lav, distinguish_day_type, f_employees, conversion_matrix, simulation_start, l, country):
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

        f_remote: pd.series
            Fraction of employees working from home per sector of NACE 64 during first 2020 Belgian COVID-19 lockdown.
            Index: NACE64 sectors

        hesitancy: pd.Series
            Normalized product of `f_remote` (fraction employees able to work remotely) and the physical proximity of workplace contacts (Pichler).
            Informs the hesitancy to stay home from work

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

        country: str
            'BE' or 'SWE'
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

        # Assign to variables
        self.age_classes = age_classes
        self.distinguish_day_type = distinguish_day_type
        self.lmc_df = lmc_df
        self.f_workplace = f_workplace
        self.f_remote = f_remote
        self.lav = lav
        self.f_employees = f_employees
        self.conversion_matrix = conversion_matrix
        self.l = l
        self.hesitancy = hesitancy
        self.country = country

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
    def __call__(self, t, M_work, M_eff, social_restrictions, economic_closures):

        # check daytype
        vacation = is_school_holiday(t, self.country)
        if self.distinguish_day_type:
            if ((t.weekday() == 5) | (t.weekday() == 6)):
                type_day = 'weekendday'
            else:
                type_day = 'weekday'
        else:
            type_day = 'average'

        # slice right matrices and convert to right size
        N_home, N_leisure_private, N_leisure_public, N_school, N_work = self.slice_matrices(type_day, vacation)

        #####################
        ## forced response ##
        #####################

        # convert economic policy from tuple to numpy array
        assert isinstance(economic_closures, tuple)
        economic_closures = 1-np.array(economic_closures, dtype=float)

        # convert the leisure_public contacts depending on the economic policy
        f_leisure_public = sum(self.lav * economic_closures)

        # zero in forced `economic_closures` corresponds to full lockdown in Belgium
        economic_closures = self.f_workplace.values + economic_closures*(1-self.f_workplace.values)

        ###############################
        ## voluntary response (work) ##
        ###############################

        assert isinstance(M_work, tuple)
        M_work = np.array(M_work, dtype=float)

        # compare to involuntary changes and take minimum as limiting
        economic_closures = np.minimum(economic_closures, M_work)

        # assert degree of school opennness
        f_school = economic_closures[np.where(self.f_workplace.index == 'P85')[0][0]]

        ##################################
        ## voluntary response (leisure) ##
        ##################################

        # public leisure contacts
        f_leisure_public = min(f_leisure_public, M_eff)

        # private leisure contacts
        f_leisure_private = min((1-social_restrictions), M_eff)

        ########################
        ## construct matrices ##
        ########################

        ## work
        # convert economic policy from NACE 64 to NACE 21
        economic_closures = np.matmul(economic_closures*self.f_employees.values, self.conversion_matrix)
        # multiply the work contacts (age, age, NACE 21) with the openness of the sectors
        N_work = N_work*economic_closures[np.newaxis, np.newaxis, :]
        # convert work contacts to (age, age, spatial_unit) using the labor market structure
        N_work = np.einsum('ijk, kl -> ijl', N_work, np.transpose(self.lmc_df.values.reshape([self.G, len(economic_closures)])))
        
        ## school
        N_school *= f_school

        ## leisure_public
        N_leisure_public *= f_leisure_public

        ## leisure_private
        N_leisure_private *= f_leisure_private

        return {'other': N_home + M_eff*(N_school + N_leisure_private + N_leisure_public), 'work': M_eff*N_work}

    def get_contacts(self, t, states, param, tau, ypsilon_work, phi_work, ypsilon_eff, phi_eff, social_restrictions, economic_closures):
        """
        Function returning the number of social contacts under sector closure and/or lockdown

        input
        =====

        tau: int/float
            half-life of the hospital load memory.
            implemented as the half-life of the exponential decay function used as weights in the computation of the exponential moving average number of hospital load

        ypsilon: int/float
            displacement parameter of the Gompertz behavioral model

        phi: int/float
            steepness parameter of the Gompertz behavioral model
        
        social_restrictions: int
            restrictions on private leisure contacts: 0: no restrictions. 1: restrictions.

        economic_closures: np.ndarray (size 63)
            closure of economic sectors (NACE 64 classification). 0: open. 1: closed.

        output
        ======

        N: dict
            Keys: "work" and "other"
            Contact matrix per spatial patch at work and in all other locations.
        """

        ############
        ## memory ##
        ############

        # get total number of hospitalisations per spatial patch per 100 K inhabitants
        I = 1e5*np.sum(states['Ih'], axis=0)/(np.sum(states['S'], axis=0) + np.sum(states['E'], axis=0) + np.sum(states['Ip'], axis=0) + np.sum(states['Ia'], axis=0) + np.sum(states['Im'], axis=0) + np.sum(states['Ih'], axis=0) + np.sum(states['R'], axis=0))
        # if timestep is less then 0.5 day from start the user wants a simulation restart
        case_tol = 31
        threshold = 3
        if ((abs((t - self.simulation_start)/timedelta(days=1)) < case_tol) & (max(I) <= threshold)):
            # re-initialize memory
            self.memory_index = [0,] 
            self.memory_values = [[I[g],] for g in range(self.G)]
            self.I_star = I 
        # determine length of timestep
        delta_t = (t - self.t)/timedelta(days=1)
        self.t = t
        # add case count to memory (RK can step backwards)
        if delta_t > 0:
            # copy values
            new_index = self.memory_index
            new_values = self.memory_values
            # append new values
            for g in range(self.G):
                new_values[g].append(I[g])
            new_index.append(new_index[-1] + delta_t)
            # subtract the new timestep
            new_index = np.array(new_index) - new_index[-1]
            # cut of values and index to not exceed memory length l
            for g in range(self.G):
                new_values[g] = list(np.array(new_values[g])[new_index >= -self.l])
            new_index = new_index[new_index >= -self.l]
            # compute exponential weights at new_index
            weights = np.exp((1/tau)*new_index)/sum(np.exp((1/tau)*new_index))
            # multiply weights with case count and sum to compute average
            I_star = np.sum(np.array(new_values)*weights, axis=1)
            # update memory
            self.memory_index = list(new_index)
            self.memory_values = new_values
            # update I_star
            self.I_star = I_star

        #######################
        ## behavioral models ##
        #######################

        # leisure and general effectivity of contacts
        M_eff = 1-self.gompertz(max(self.I_star), ypsilon_eff, phi_eff)

        # voluntary switch to telework or absenteism
        M_work = 1-self.gompertz(max(self.I_star), ypsilon_work, (phi_work*self.hesitancy).values)
        
        ##############
        ## policies ##
        ##############

        t_start_lockdown = datetime(2020, 3, 15) # start of lockdown
        t_end_lockdown = datetime(2020, 5, 7)

        if t < t_start_lockdown:
            return self.__call__(t, tuple(M_work), M_eff, 0, tuple(np.zeros(63, dtype=float)))
        elif t_start_lockdown < t < t_end_lockdown:
            policy_old = self.__call__(t, tuple(M_work), M_eff, 0, tuple(np.zeros(63, dtype=float)))
            policy_new = self.__call__(t, tuple(M_work), M_eff, social_restrictions, tuple(economic_closures))
            return {'other': ramp_fun(t, t_start_lockdown, 7, policy_old['other'], policy_new['other']),
                    'work': ramp_fun(t, t_start_lockdown, 7, policy_old['work'], policy_new['work'])}
        else:
            return self.__call__(t, tuple(M_work), M_eff, 0, tuple(np.zeros(63, dtype=float)))

    @staticmethod
    def gompertz(x, a, b):
        """
        A Gompertz behavioral model

        input
        =====
        x: float or np.ndarray
            input value. range: [-infinity --> infinity]

        a: float
            displacement along x-axis + y(0). higher values displace to the right and lower y(0) to zero. (if alpha > 5 then y(0) approx. 0)
        
        b: float
            steepness parameter.

        output
        ======

        y: float or np.ndarray
            output value. range: [0,1]

        """
        return np.exp(-a*np.exp(-b*x))

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