# import packages used
import numpy as np
from scipy import interpolate
import scipy.optimize as optimize
from tools import nonlinspace, gauss_hermite

def util(c,par):
    return (c**(1.0-par.rho))/(1.0-par.rho)

def marg_util(c,par):
    return c**(-par.rho)

def setup():

    # 1. setup specifications in class. 
    class par: pass
    par.beta = 0.98
    par.rho = 0.5
    par.R = 1.0/par.beta
    par.sigma = 0.2
    par.mu = 0
    par.M = 10
    par.T = 10
    
    # 2. gauss Hermite weights and poins
    par.num_shocks = 5
    x,w = gauss_hermite(par.num_shocks)
    par.eps = np.exp(par.sigma*np.sqrt(2)*x)
    par.eps_w = w/np.sqrt(np.pi)
    
    # 3. simulation parameters
    par.simN = 10000
    par.M_ini = 1.5
    
    # 4. grid
    par.num_M = 100
    par.grid_M = nonlinspace(1.0e-6,par.M,par.num_M,1.1) # same as np.linspace just with unequal spacing
    
    # 5. dimension of value function space
    par.dim = [par.num_M,par.T]
    
    return par

def solve_ti(par):

    # 1. initialize solution class
    class sol: pass
    sol.C = np.zeros(par.dim) + np.nan
    
    # 2. last period, consume everything
    sol.C[:,par.T-1] = par.grid_M
    
    # 3. loop over periods
    for t in reversed(range(0,par.T-1)):  # T-2, T-3, ..., 0 
    
            # a. picking some arbitrary small starting value
            x0 = np.ones(par.num_M)*1.0e-7 
            
            # b. define the objective function
            obj_fun = lambda x: euler_error_func(x,t,par,sol)
            
            # c. find roots
            res = optimize.root(obj_fun, x0)
            x1 = res.x
            
            # d. Handle corner solutions
            x1[x1>par.grid_M] = par.grid_M[x1>par.grid_M] # set consumption to assets (consume everything)
            
            # e. final solution
            sol.C[:,t] = x1
        
    return sol

def euler_error_func(c,t,par,sol):
    """
    Function that computes Euler error given consumption in period t and future policies
    """
    
    # 1. find next period's assets

    # FILL IN. Delete "None".
    # Hint: create a matrix with state grid points as rows and add the different shocks as columns
    m_next = None




    # 2. interpolate next period's consumption
    interp = interpolate.interp1d(par.grid_M,sol.C[:, t+1], bounds_error=False, fill_value = "extrapolate") 
    c_next = interp(m_next)

    # 3. calculate next period expected marginal utility
    EMU_next = np.sum(par.eps_w[None, :]*marg_util(c_next ,par), axis=1)
    
    # 4. calculate current period marginal utility
    MU_now = marg_util(c,par) 

    # 5. calculate Euler error

    # FILL IN. Delete "None".
    euler_error = None




    return euler_error

def simulate (par,sol):
    
    # 1. initialize
    class sim: pass
    sim.M = par.M_ini*np.ones((par.simN, par.T))
    sim.C = np.zeros((par.simN, par.T)) + np.nan
    
    # 2. Simulate
    np.random.seed(2026) 
    for t in range(par.T):
        
        # a. interpolate consumption given current wealth
        interp = interpolate.interp1d(par.grid_M,sol.C[:,t], bounds_error=False, fill_value = "extrapolate") 
        sim.C[:, t] = interp(sim.M[:, t])
    
        # b. handle state transition
        if t<par.T-1:  # if not last period

            # i. draw random shock
            logY = np.random.normal(par.mu, par.sigma, par.simN)
            Y = np.exp(logY)
            
            # ii. compute next period assets
            A = sim.M[:, t]-sim.C[:, t]
            sim.M[:, t+1] = par.R*A + Y
     
    return sim