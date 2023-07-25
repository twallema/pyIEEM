import numpy as np
from pySODM.models.base import ODE

class SIR(ODE):
    """
    An extendable base class for the disease model
    """

    # state variables and parameters
    states = ['S', 'I', 'R',]
    parameters = ['beta', 'gamma', 'N', 'G']
    dimensions = ['age_class', 'spatial_unit']

    @staticmethod
    def integrate(t, S, I, R, beta, gamma, N, G):

        # compute total population
        T = S + I + R
        
        # compute work populations
        T_work = np.matmul(T, G)
        S_work = np.matmul(S, G)
        I_work = np.matmul(I, G)

        # compute infection pressure in home patch and work patch
        IP_work = beta*np.einsum('ij, jki -> ki', np.transpose(I_work/T_work), N['work'])
        IP_other = beta*np.einsum('ij, jki -> ki', np.transpose(I/T), N['other'])

        # compute number of infections
        n_inf = S * IP_other + S_work * IP_work

        # equations
        dS = - n_inf
        dI = n_inf - 1/gamma*I
        dR = 1/gamma*I

        return dS, dI, dR
