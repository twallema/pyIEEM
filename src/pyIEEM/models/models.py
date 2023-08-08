import numpy as np
from pySODM.models.base import ODE

class SIR(ODE):
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
