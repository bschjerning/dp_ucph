# import packages used
import numpy as np
from scipy import interpolate
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import tools

def util(c,par):
    return (c**(1.0-par.rho))/(1.0-par.rho)

def marg_util(c,par):
    return c**(-par.rho)

def inv_marg_util(u,par):
    return u**(-1/par.rho)

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

    #4. End of period assets - used for EGM-solution
    par.grid_a = nonlinspace(0 + 1e-8,par.M,par.num_M,1.1)

    # Dimension of value function space
    par.dim = [par.num_M,par.T]
    
    return par

def EGM_loop (sol,t,par):

    interp = interpolate.interp1d(sol.M[:,t+1],sol.C[:,t+1], bounds_error=False, fill_value = "extrapolate") 
    for i_a,a in enumerate(par.grid_a):

        # Futute m and c
        m_next = par.R * a + par.eps
        c_next = interp(m_next)
        # Future marginal utility
        EU_next = np.sum(par.eps_w*marg_util(c_next,par))

        # Currect C and m
        sol.C[i_a+1,t]= inv_marg_util(par.R * par.beta * EU_next, par)
        sol.M[i_a+1,t]= sol.C[i_a+1,t] + a

    return sol


def EGM_vectorized (sol,t,par):

    interp = interpolate.interp1d(sol.M[:,t+1],sol.C[:,t+1], bounds_error=False, fill_value = "extrapolate")

    # weights = np.tile(par.eps_w, par.num_M)
    nodes = np.tile(par.eps, par.num_M)
    a = np.repeat(par.grid_a, par.num_shocks)
    m_next  = par.R * a + nodes
    c_next = interp(m_next)
    marg_u_next = marg_util(c_next,par)
    marg_u_next =  np.reshape(marg_u_next, (par.num_M, par.num_shocks))
    EU_next = np.sum(par.eps_w * marg_u_next, axis = 1)
    # Currect C and m
    sol.C[1:,t]= inv_marg_util(par.R * par.beta * EU_next, par)
    sol.M[1:,t]= sol.C[1:,t] + par.grid_a
    return sol


def solve_EGM(par, vector = False):
     # initialize solution class
    class sol: pass
    shape = [par.num_M+1, par.T]
    sol.C = np.nan + np.zeros(shape)
    sol.M = np.nan + np.zeros(shape)
    # Last period, consume everything
    sol.M[:,par.T-1] = nonlinspace(0,par.M,par.num_M+1,1.1)
    sol.C[:,par.T-1]= sol.M[:,par.T-1].copy()

    # Loop over periods
    for t in range(par.T-2, -1, -1):  #from period T-2, until period 0, backwards
        # sol = EGM_loop(sol, t, par) 
        if vector == True:
            sol = EGM_vectorized(sol, t, par)
        else:
            sol = EGM_loop(sol, t, par)
        # add zero consumption
        sol.M[0,t] = 0
        sol.C[0,t] = 0
    return sol

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
        interp = interpolate.interp1d(sol.M[:,t],sol.C[:,t], bounds_error=False, fill_value = "extrapolate") 
        sim.C[:,t] = interp(sim.M[:,t])  # Find consumption given state
    
        if t<par.T-1:  # if not last period
            logY = np.random.normal(par.mu,par.sigma,par.simN)  # Draw random number from the normal distirbution
            Y = np.exp(logY)
            A = sim.M[:,t]-sim.C[:,t]
        
            sim.M[:,t+1] = par.R*A + Y # The state in the following period
            
     
    return sim