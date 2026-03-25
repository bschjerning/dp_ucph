# Import package and module
import numpy as np
import utility as util
import tools
from scipy import interpolate


def EGM(sol,t,par):

    if par.vec:
        sol = EGM_vec(sol,t,par) 

    else:
        sol = EGM_loop(sol,t,par) 

    return sol

def EGM_loop(sol,t,par):

    for i_a,a in enumerate(par.grid_a[t,:]):

        # FILL IN. Hint: Same procedure as in 02_EGM (last week exercises)

        if t+1<= par.Tr:
            fac = par.G*par.L[t]*par.psi_vec # factor to normalize with



        else:
            fac = par.G*par.L[t]




        c_pol = None # delete None and fill in.
        m_implied = None




        # current c and m (i_a+1 as we save the first index point to handle the credit constraint region)
        sol.c[t,i_a+1] = c_pol
        sol.m[t,i_a+1] = m_implied

    return sol

def EGM_vec (sol,t,par):

    # FILL IN.



    return sol
