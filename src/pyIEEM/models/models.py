import numpy as np
from pySODM.models.base import ODE

class epidemic_model(ODE):
    """
    the COVID-19 disease model
    """

    # state variables and parameters
    states = ['S', 'E', 'Ip', 'Ia', 'Im', 'Ih', 'R', 'D', 'Hin']
    parameters = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 's', 'a', 'h', 'm', 'N', 'G']
    dimensions = ['age_class', 'spatial_unit']

    @staticmethod
    def integrate(t, S, E, Ip, Ia, Im, Ih, R, D, Hin, alpha, beta, gamma, delta, epsilon, zeta, s, a, h, m, N, G):

        # compute total population
        T = S + E + Ip + Ia + Im + Ih + R + D
        
        # compute work populations
        T_work = np.matmul(T, G)
        S_work = np.matmul(S, G)
        Ip_work = np.matmul(Ip, G)
        Ia_work = np.matmul(Ia, G)
        Im_work = np.matmul(Im, G)

        # compute infection pressure in home patch and work patch
        IP_work = s*beta*np.einsum('ij, jki -> ki', np.transpose((Ip_work + Ia_work + Im_work)/T_work), N['work'])
        IP_other = s*beta*np.einsum('ij, jki -> ki', np.transpose((Ip + Ia + Im)/T), N['other'])

        # compute number of infections
        n_inf = S * IP_other + S_work * IP_work

        # model equations
        dS = - n_inf + (1/zeta)*R
        dE = n_inf - (1/alpha)*E
        dIp = (1/alpha)*E - (1/gamma)*Ip
        dIa = a*(1/gamma)*Ip - (1/delta)*Ia
        dIm = (1-a)*(1/gamma)*Ip - (1/delta)*Im
        dIh = h*(1/delta)*Im - (1/epsilon)*Ih
        dR = (1/delta)*Ia + (1-h)*(1/delta)*Im + (1-m)*(1/epsilon)*Ih - (1/zeta)*R
        dD = m*(1/epsilon)*Ih

        # derivative states
        dHin = h*(1/delta)*Im - Hin

        return dS, dE, dIp, dIa, dIm, dIh, dR, dD, dHin

# All NaN slices in np.nanmin() return a RunTimeWarning
import warnings
warnings.filterwarnings("ignore")

class epinomic_model(ODE):
    """
    the coupled COVID-19 - economic model
    """

    # states
    states_epi = ['S','E','Ip','Ia','Im','Ih','R','D','Hin']
    states_eco = ['x','c', 'c_desired', 'f', 'd', 'l','O', 'St']
    states = states_epi + states_eco
    # parameters
    parameters_epi = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 's', 'a', 'h', 'm', 'N', 'G']
    parameters_eco = ['x_0', 'c_0', 'f_0', 'l_0', 'C', 'St_0', 'delta_S', 'eta', 'iota_F','iota_H', 'A', 'prodfunc', 'kappa_S', 'kappa_D', 'kappa_F']
    parameters = parameters_epi + parameters_eco
    # dimensions
    dimensions = ['age_class', 'spatial_unit', 'NACE64', 'NACE64_star']
    dimensions_per_state = [
        ['age_class', 'spatial_unit'], ['age_class', 'spatial_unit'], ['age_class', 'spatial_unit'], ['age_class', 'spatial_unit'], ['age_class', 'spatial_unit'], ['age_class', 'spatial_unit'], ['age_class', 'spatial_unit'], ['age_class', 'spatial_unit'], ['age_class', 'spatial_unit'],
        ['NACE64',], ['NACE64',], ['NACE64',], ['NACE64',], ['NACE64',], ['NACE64',], ['NACE64',], ['NACE64','NACE64_star']
    ]

    @staticmethod
    def integrate(t, S, E, Ip, Ia, Im, Ih, R, D, Hin, x, c, c_desired, f, d, l, O, St,
                    alpha, beta, gamma, delta, epsilon, zeta, s, a, h, m, N, G, 
                    x_0, c_0, f_0, l_0, C, St_0, delta_S, eta, iota_F, iota_H, A, prodfunc, kappa_S, kappa_D, kappa_F):

        #######################
        ## epidemic dynamics ##
        #######################

        # compute total population
        T = S + E + Ip + Ia + Im + Ih + R + D
        
        # compute work populations
        T_work = np.matmul(T, G)
        S_work = np.matmul(S, G)
        Ip_work = np.matmul(Ip, G)
        Ia_work = np.matmul(Ia, G)
        Im_work = np.matmul(Im, G)

        # compute infection pressure in home patch and work patch
        IP_work = s*beta*np.einsum('ij, jki -> ki', np.transpose((Ip_work + Ia_work + Im_work)/T_work), N['work'])
        IP_other = s*beta*np.einsum('ij, jki -> ki', np.transpose((Ip + Ia + Im)/T), N['other'])

        # compute number of infections
        n_inf = S * IP_other + S_work * IP_work

        # model equations
        dS = - n_inf + (1/zeta)*R
        dE = n_inf - (1/alpha)*E
        dIp = (1/alpha)*E - (1/gamma)*Ip
        dIa = a*(1/gamma)*Ip - (1/delta)*Ia
        dIm = (1-a)*(1/gamma)*Ip - (1/delta)*Im
        dIh = h*(1/delta)*Im - (1/epsilon)*Ih
        dR = (1/delta)*Ia + (1-h)*(1/delta)*Im + (1-m)*(1/epsilon)*Ih - (1/zeta)*R
        dD = m*(1/epsilon)*Ih

        # derivative states
        dHin = h*(1/delta)*Im - Hin

        #######################
        ## economic dynamics ##
        #######################

        # 1. Update exogeneous demand with shock vector
        # ---------------------------------------------
        f_desired = (1-kappa_F)*f_0

        # 3. Compute productive capacity under labor constraints
        # ------------------------------------------------------
        x_cap = calc_labor_restriction(x_0, l_0, l)

        # 4. Compute productive capacity under input constraints
        # ------------------------------------------------------
        x_inp = calc_input_restriction(St, A, C, x_0, prodfunc)

        # 5. Compute total consumer demand
        # --------------------------------

        # Compute aggregate household demand shock
        theta = household_preference_shock(kappa_D, c_0/sum(c_0))
        # compute shocked consumer preference vector
        epsilon_D = aggregate_demand_shock(kappa_D, c_0/sum(c_0), delta_S)
        # compute shocked household demand
        c_desired_new = (1 - epsilon_D) * theta * sum(c_0)

        # 6. Compute B2B demand
        # ---------------------   
        O_desired = calc_intermediate_demand(d, St, A, St_0, eta) # 2D

        # 7. Compute total demand
        # -----------------------
        d_new = calc_total_demand(O_desired, c_desired_new, f_desired)

        # 8. Leontief production function with critical inputs
        # ----------------------------------------------------
        x_new = leontief(x_cap, x_inp, d_new)

        # 9. Perform rationing
        # --------------------
        O_new, c_new, f_new = rationing(x_new, d_new, O_desired, c_desired_new, f_desired)

        # 10. Update inventories
        # ----------------------
        St_new = inventory_updating(St, O_new, x_new, A)

        # 11. Hire/fire workers
        # ---------------------
        l_new = hiring_firing(l, l_0, x_0, x_inp, x_cap, d_new, iota_F, iota_H, kappa_S)

        # 12. Convert order matrix to total order per sector (2D --> 1D)
        # --------------------------------------------------------------
        O_new = np.sum(O_new,axis=1)

        return dS, dE, dIp, dIa, dIm, dIh, dR, dD, dHin, x_new-x, c_new-c, c_desired_new-c_desired, f-f_new, d_new-d, l_new-l, O_new-O, St_new-St,


def calc_labor_restriction(x_0,l_0,l_t):
    """
    A function to compute sector output with the available labor force.

    Parameters
    ----------
    x_0 : np.array
        sector output under business-as-usual (in M€/d)
    l_0 : np.array
        number of employees per sector under business-as-usual
    l_t : np.array
        number of employees at time t

    Returns
    -------
    x_t : np.array
        sector output at time t (in M€/d)
    """
    return (l_t/l_0)*x_0

def calc_input_restriction(S_t, A, C, x_0, prodfunc='half_critical'):
    """
    A function to compute sector output under supply bottlenecks.

    Parameters
    ----------
    S_t : np.array
        stock matrix at time t    
    A : np.array
        matrix of technical coefficients
    C : np.array
        matrix of critical inputs

    Returns
    -------
    x_t : np.array
        sector output at time t (in M€/d)
    """

    # Pre-allocate sector output at time t
    x_t = np.zeros(A.shape[0])
    # Loop over all sectors
    if prodfunc == 'linear':
        for i in range(A.shape[0]):
            x_t[i] = np.sum(S_t[:,i])/np.sum(A[:,i])
    elif prodfunc == 'weakly_critical':
        for i in range(A.shape[0]):
            critical = list(np.where(C[:,i] == 1)[0])
            x_t[i] = np.nanmin(S_t[critical,i]/A[critical,i])
            if np.isnan(x_t[i]):
                x_t[i]=np.inf
    elif prodfunc == 'half_critical':
        cond_1 = np.zeros(A.shape[0])
        cond_2 = np.zeros(A.shape[0])
        for i in range(A.shape[0]):
            critical = list(np.where(C[:,i] == 1)[0])
            important = list(np.where(C[:,i] == 0.5)[0])
            cond_1[i] = np.nanmin(S_t[critical,i]/A[critical,i])
            if len(important) == 0:
                x_t[i] = cond_1[i]
            else:
                cond_2[i] = np.nanmin(0.5*(np.array(S_t[important,i]/A[important,i]) + x_0[i]))
                x_t[i] = np.nanmin(np.array([cond_1[i], cond_2[i]]))
            if np.isnan(x_t[i]):
                x_t[i]=np.inf
    elif prodfunc == 'strongly_critical':
        for i in range(A.shape[0]):
            critical = list(np.where(C[:,i] == 1)[0])
            important = list(np.where(C[:,i] == 0.5)[0])
            x_t[i] = np.nanmin(S_t[critical+important,i]/A[critical+important,i])
            if np.isnan(x_t[i]):
                x_t[i]=np.inf
    elif prodfunc == 'leontief':
        for i in range(A.shape[0]):
            x_t[i] = np.nanmin(S_t[:,i]/A[:,i])
            if np.isnan(x_t[i]):
                x_t[i]=np.inf    
    return x_t

def household_preference_shock(epsilon_D, theta_0):
    """
    A function to return the preference of households for the output of a certain sector

    Parameters
    ----------
    epsilon_D : np.array
        sectoral household demand shock
    theta_0 : int
        household preference under business-as-usual (absence of shock epsilon)

    Returns
    -------
    theta : np.array
        household consumption preference vector
    """

    theta=np.zeros(epsilon_D.shape[0])
    for i in range(epsilon_D.shape[0]):
        theta[i] = (1-epsilon_D[i])*theta_0[i]/(sum((1-epsilon_D)*theta_0))
    return theta

def aggregate_demand_shock(mu_D, theta_0, delta_S):
    """
    A function to return the aggregate household demand shock.

    Parameters
    ----------

    mu_D: np.array
        household demand shock

    theta_0: np.array
        household preference coefficient. denotes share of good i in aggregate household demand

    delta_S: float
        savings rate of households (delta_S = 1; households save all money they are not spending due to shock)

    Returns
    -------
    epsilon_D: float
        aggregate household demand shock
    """
    return delta_S*(1-sum((1-mu_D)*theta_0))

def calc_intermediate_demand(d_previous, St, A, St_0, iota):
    """
    A function to calculate the intermediate demand between sectors (B2B demand).
    = Restocking function (1st order system with time constant iota)

    Parameters
    ----------
    d_previous : np.array
        total demand (per sector) at time t -1
    St : np.array
        stock matrix at time t
    A : np.array
        matrix of technical coefficients
    St_0 : np.array
        desired stock matrix
    tau : int
        restock speed

    Returns
    -------
    O : np.array
        matrix of B2B orders
    """
    O = np.zeros([A.shape[0],A.shape[0]])
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            O[i,j] = A[i,j]*d_previous[j] + (1/iota)*(St_0[i,j] - St[i,j])
    return O

def calc_total_demand(O, c_t, f_t):
    """
    A function to calculate the total demand for the output of every sector

    Parameters
    ----------
    O : np.array
        matrix of B2B orders
    c_t : np.array
        household demand
    f_t : np.array
        other demand

    Returns
    -------
    d_t : np.array
        total demand
    """
    return np.sum(O,axis=1) + c_t + f_t


def rationing(x_t,d_t,O,c_t,f_t):
    """
    A function to ration the output if output doesn't meet demand.
    No prioritizing between B2B restocking, households and others (government/exports) is performed.

    Parameters
    ----------
    x_t : np.array
        total output of sector i
    d_t : np.array
        total demand for the output of sector i
    O : np.array
        matrix of B2B orders
    c_t : np.array
        total household demand for the output of sector i
    f_t : np.array
        total other demand for the output of sector i

    Returns
    -------
    Z_t : np.array
        fraction r of B2B orders received
    r*c_t : np.array
        fraction r of household demand met
    r*f_t : np.array
        fraction r of other demand met
    """

    scheme='proportional_strict'

    if scheme == 'proportional_strict':
        r = x_t/d_t
        r[np.where(r > 1)] = 1
        Z_t = np.zeros([O.shape[0],O.shape[0]])
        for i in range(O.shape[0]):
                Z_t[i,:] = O[i,:]*r[i]
        return Z_t,r*c_t,r*f_t

    elif scheme == 'proportional_priority_B2B':
        # B2B priority
        r = x_t/np.sum(O, axis=1)
        r[np.where(r > 1)] = 1
        Z_t = np.zeros([O.shape[0],O.shape[0]])
        for i in range(O.shape[0]):
                Z_t[i,:] = O[i,:]*r[i]
        # Proportional rationing
        l = x_t - np.sum(Z_t, axis=1)
        l[np.where(l < 0)] = 0
        r = l/(c_t + f_t)
        r[np.where(r > 1)] = 1
        return Z_t, r*c_t, r*f_t

    elif scheme == 'random_priority_B2B':
        # Why the f*@ck is this necessary?
        x_t_copy = x_t.copy()
        Z_t = np.zeros([O.shape[0],O.shape[0]])
        # Generate a random priority vector
        priority = list(range(O.shape[0]))
        for i in range(O.shape[0]):
            np.random.shuffle(priority)
            for j in range(O.shape[0]):
                # Get sector index of current priority
                j = priority.index(j)
                # Check if industry i produces enough to satisfy the demand of sector j
                r = x_t_copy[i]/O[i,j]
                if r > 1:
                    r=1
                if ((np.isinf(r))|(np.isnan(r))|(r < 0)):
                    r=0
                Z_t[i,j] = r*O[i,j]
                x_t_copy[i] -= Z_t[i,j]
        # Ration rest
        r = x_t_copy/(c_t + f_t)
        r[np.where(r > 1)] = 1
        return Z_t, r*c_t, r*f_t

    elif scheme == 'largest_first_priority_B2B':
        # Why the f*@ck is this necessary?
        x_t_copy = x_t.copy()
        Z_t = np.zeros([O.shape[0],O.shape[0]])
        for i in range(O.shape[0]):
            customer_value = list(O[i,:])
            customer_value.sort(reverse=True)
            for value in customer_value:
                # Get sector index of current priority
                j = list(O[i,:]).index(value)
                # Check if industry i produces enough to satisfy the demand of sector j
                r = x_t_copy[i]/O[i,j]
                if r > 1:
                    r=1
                if ((np.isinf(r))|(np.isnan(r))|(r < 0)):
                    r=0
                Z_t[i,j] = r*O[i,j]
                x_t_copy[i] = x_t_copy[i] - Z_t[i,j]
        # Ration rest
        r = x_t_copy/(c_t + f_t)
        r[np.where(r > 1)] = 1
        return Z_t, r*c_t, r*f_t

def leontief(x_t_labor, x_t_input, d_t):
    """
    An implementation of the Leontief production function.

    Parameters
    ----------
    x_t_labor : np.array
        sectoral output at time t under labor constraints
    x_t_input : np.array
        sectoral output at time t under input constraints
    d_t : np.array
        total demand at time t

    Returns
    -------
    x_t : np.array
        sector output at time t (in M€)
    """
    return np.amin([x_t_labor, x_t_input, d_t],axis = 0)

def inventory_updating(St_old, Z_t, x_t, A):
    """
    A function to update the inventory.

    Parameters
    ----------
    St_old : np.array
        Stock matrix at time t
    Z_t : np.array
        Orders received at time t
    x_t : np.array
        Total output produced at time t
    A : np.array
        Matrix of technical coefficients (input need per unit output)

    Returns
    -------
    S_new : np.array
        
    """
    St_new = np.zeros([St_old.shape[0],St_old.shape[0]])
    for i in range(St_old.shape[0]):
        for j in range(St_old.shape[0]):
            St_new[i,j] = St_old[i,j] + Z_t[i,j] - A[i,j]*x_t[j]
    St_new[np.where(St_new < 0)] = 0
    return St_new


def hiring_firing(l_old, l_0, x_0, x_t_input, x_t_labor, d_t, kappa_F, kappa_H, mu_S):
    """
    A function to update the labor income. (= proxy for size of workforce)

    Parameters
    ----------
    l_old : np.array
        labor income at time t
    l_0 : np.array
        labor income during business-as-usual
    x_0 : np.array
        sector output during business-as-usual
    x_t_input : np.array
        maximum output at time t due to supply bottlenecks
    x_t_labor : np.array
        maximum output at time t due to labor reduction
    d_t : np.array
        total demand at time t
    kappa_F : float
        number of days needed to fire a worker
    kappa_H : float
        number of days needed to hire a worker
    mu_S : np.array
        Labor supply shock

    Returns
    -------
    l_new : np.array
        labor income at time t + 1
        
    """
    # Normal hiring/firing procedure
    delta_l = (l_0/x_0)*(np.minimum(x_t_input,d_t)-x_t_labor)
    l_new=np.zeros([delta_l.shape[0]])
    for i in range(delta_l.shape[0]):
        if delta_l[i] > 0:
            l_new[i] = l_old[i] + 1/kappa_H*delta_l[i]
        elif delta_l[i] <= 0:
            l_new[i] = l_old[i] + 1/kappa_F*delta_l[i]
    l_new=np.expand_dims(l_new,axis=1)
    l_0=np.expand_dims(l_0,axis=1)
    mu_S=np.expand_dims(mu_S,axis=1)
    # Labor force reduction due to lockdown
    l_new[np.greater(l_new,(1-mu_S)*l_0)] =  ((1-mu_S)*l_0)[np.greater(l_new,(1-mu_S)*l_0)]
    return l_new[:,0]