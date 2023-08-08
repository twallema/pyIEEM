import numpy as np
from pySODM.models.base import ODE

class epidemic_model(ODE):
    """
    An extendable base class for the disease model
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

class epinomic_model(ODE):
    """
    My epinomic model (C)
    """

    # states
    states_epi = ['S','E','Ip','Ia','Im','Ih','R','D','Hin']
    states_eco = ['x','c', 'c_desired','f', 'd', 'l','O', 'St']
    states = states_epi + states_eco
    # parameters
    parameters_epi = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 's', 'a', 'h', 'm', 'N', 'G']
    parameters_eco = ['x_0', 'c_0', 'f_0', 'l_0', 'IO', 'O_j', 'n', 'on_site', 'C', 'St_0','b','eta','delta_S','theta','iota','kappa_F','kappa_H', 'A', 'prodfunc']
    parameters = parameters_epi + parameters_eco
    # dimensions
    dimensions = ['age_class', 'spatial_unit', 'NACE64', 'NACE64_star']
    dimensions_per_state = [len(states_epi)*['age_class', 'spatial_unit'], (len(states_eco)-1)*['NACE64',] + ['NACE64','NACE64_star']]

    @staticmethod
    def integrate(t, S, E, Ip, Ia, Im, Ih, R, D, Hin, x, c, c_desired, f, d, l, O, St,
                    alpha, beta, gamma, delta, epsilon, zeta, s, a, h, m, N, G, 
                    x_0, c_0, f_0, l_0, IO, O_j, n, on_site, C, St_0, b, eta, delta_S, theta, iota, appa_F, kappa_H, A, prodfunc):

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

        x_new = x
        c_new = c
        c_desired_new = c_desired
        f_new = f
        d_new = D
        l_new = l
        O_new = O
        St_new = St

        return dS, dE, dIp, dIa, dIm, dIh, dR, dD, dHin, x_new-x, c_new-c, c_desired_new-c_desired, f-f_new, d_new-d, l_new-l, O_new-O, St_new-St,
