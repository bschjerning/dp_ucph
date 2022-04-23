# import packages used
import numpy as np
from scipy import interpolate
import scipy.optimize as optimize
import matplotlib.pyplot as plt

def util(c,par):
    return (c**(1.0-par.rho))/(1.0-par.rho)

def marg_util(c,par):
    return c**(-par.rho)

def setup():
    # Setup specifications in class. 
    class par: pass
    par.beta = 0.98
    par.rho = 0.5
    par.R = 1.0/par.beta
    par.sigma = 0.2
    par.mu = 0
    par.M = 10
    par.T = 10
    
    # Gauss Hermite weights and poins
    par.num_shocks = 5
    x,w = gauss_hermite(par.num_shocks)
    par.eps = np.exp(par.sigma*np.sqrt(2)*x)
    par.eps_w = w/np.sqrt(np.pi)
    
    # Simulation parameters
    par.simN = 10000
    par.M_ini = 1.5
    
    # Grid
    par.num_M = 100
    par.grid_M = nonlinspace(1.0e-6,par.M,par.num_M,1.1) # same as np.linspace just with unequal spacing
    
    # Dimension of value function space
    par.dim = [par.num_M,par.T]
    
    return par

def solve_ti(par):
     # initialize solution class
    class sol: pass
    sol.C = np.zeros(par.dim)
    
    # Last period, consume everything
    sol.C[:,par.T-1] = par.grid_M
    
    # Loop over periods
    for t in range(par.T-2, -1, -1):  #from period T-2, until period 0, backwards 
    
            x0 = np.ones(par.num_M)*1.0e-7 # Picking some arbitrary small starting value
            
            obj_fun = lambda x: euler_error_func(x,t,par,sol)
            
            res = optimize.fsolve(obj_fun, x0)
            
            # corner solution
            I = res>par.grid_M
            res[I] = par.grid_M[I]
            
            # final solution
            sol.C[:,t] = res
            
        
    return sol

def euler_error_func(x,t,par,sol):
    
    c = x
    
    m_next = par.R*(par.grid_M - c)[:,np.newaxis] + par.eps[np.newaxis,:] # creating a matrix with state grid points as rows and different shocks as columns

    interp = interpolate.interp1d(par.grid_M,sol.C[:,t+1], bounds_error=False, fill_value = "extrapolate") 

    c_next = interp(m_next)

    EU_next = np.sum(par.eps_w[np.newaxis,:]*marg_util(c_next,par), axis=1) # Expected marginal utility next period
    
    U_now = marg_util(c,par)    # Marginal utility this period

    euler_error = U_now-par.beta*par.R*EU_next

    return euler_error


def gauss_hermite(n):

    # a. calculations
    i = np.arange(1,n)
    a = np.sqrt(i/2)
    CM = np.diag(a,1) + np.diag(a,-1)
    L,V = np.linalg.eig(CM)
    I = L.argsort()
    V = V[:,I].T

    # b. nodes and weights
    x = L[I]
    w = np.sqrt(np.pi)*V[:,0]**2

    return x,w

def nonlinspace(x_min, x_max, n, phi):
    """ like np.linspace between with unequal spacing
    phi = 1 -> eqaul spacing
    phi up -> more points closer to minimum
    """
    assert x_max > x_min
    assert n >= 2
    assert phi >= 1
 
    # 1. recursion
    y = np.empty(n)
 
    y[0] = x_min
    for i in range(1, n):
        y[i] = y[i-1] + (x_max-y[i-1]) / (n-i)**phi
    
    # 3. assert increaing
    assert np.all(np.diff(y) > 0)
 
    return y

def simulate (par,sol):
    
    # Initialize
    class sim: pass
    dim = (par.simN,par.T)
    sim.M = par.M_ini*np.ones(dim)
    sim.C = np.nan +np.zeros(dim)
    np.random.seed(2022)

    # Simulate 
    for t in range(par.T):
        interp = interpolate.interp1d(par.grid_M,sol.C[:,t], bounds_error=False, fill_value = "extrapolate") 
        sim.C[:,t] = interp(sim.M[:,t])  # Find consumption given state
    
        if t<par.T-1:  # if not last period
            logY = np.random.normal(par.mu,par.sigma,par.simN)  # Draw random number from the normal distirbution
            Y = np.exp(logY)
            A = sim.M[:,t]-sim.C[:,t]
        
            sim.M[:,t+1] = par.R*A + Y # The state in the following period
            
     
    return sim