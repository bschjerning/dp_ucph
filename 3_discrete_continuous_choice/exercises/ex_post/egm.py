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

            ### SOLUTION ###
            w = par.w
            xi = par.xi_vec
            m_plus = (1/fac)*par.R*a + xi
            c_plus = tools.interp_linear_1d(sol.m[t+1, :], sol.c[t+1, :], m_plus)
            ### SOLUTION ###

        else:
            fac = par.G*par.L[t]

            ### SOLUTION ###
            w = 1
            xi = 1
            m_plus = (1/fac)*par.R*a + xi
            c_plus = tools.interp_linear_1d_scalar(sol.m[t+1, :], sol.c[t+1, :], m_plus)
            ### SOLUTION ###

        c_pol = None # delete None and fill in.
        m_implied = None

        ### SOLUTION ###
        marg_u_plus = util.marg_util(fac*c_plus,par)
        avg_marg_u_plus = np.sum(w*marg_u_plus)
        c_pol = util.inv_marg_util(par.beta*par.R*avg_marg_u_plus,par)
        m_implied = a + c_pol
        ### SOLUTION

        # current c and m (i_a+1 as we save the first index point to handle the credit constraint region)
        sol.c[t,i_a+1] = c_pol
        sol.m[t,i_a+1] = m_implied

    return sol

def EGM_vec (sol,t,par):

    # FILL IN.

    ### SOLUTION ###
    if t+1 <= par.Tr: 
        fac = np.tile(par.G*par.L[t]*par.psi_vec, par.Na) 
        xi = np.tile(par.xi_vec,par.Na)
        a = np.repeat(par.grid_a[t],par.Nshocks) 

        w = np.tile(par.w,(par.Na,1))
        dim = par.Nshocks
        
    else:
        fac = par.G*par.L[t]*np.ones((par.Na))
        xi = np.ones((par.Na))
        a = par.grid_a[t,:]
            
        w = np.ones((par.Na,1))
        dim = 1

    m_plus = (1/fac)*par.R*a+xi
    c_plus = tools.interp_linear_1d(sol.m[t+1, :], sol.c[t+1, :], m_plus)
    marg_u_plus = util.marg_util(fac*c_plus,par)
    marg_u_plus = np.reshape(marg_u_plus,(par.Na,dim))
    avg_marg_u_plus = np.sum(w*marg_u_plus,1)
    sol.c[t, 1:] = util.inv_marg_util(par.beta*par.R*avg_marg_u_plus,par)
    sol.m[t, 1:] = par.grid_a[t, :] + sol.c[t, 1:]
    ### SOLUTION ###

    return sol
