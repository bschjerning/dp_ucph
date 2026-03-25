# import packages used
import numpy as np
from scipy import interpolate
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from tools import gauss_hermite, nonlinspace

def util(c,par):
    return (c**(1.0-par.rho))/(1.0-par.rho)

def marg_util(c,par):
    return c**(-par.rho)

def inv_marg_util(u,par):
    return u**(-1/par.rho)

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
    
    # 2. Gauss Hermite weights and points
    par.num_shocks = 5
    x,w = gauss_hermite(par.num_shocks)
    par.eps = np.exp(par.sigma*np.sqrt(2)*x)
    par.eps_w = w/np.sqrt(np.pi)
    
    # 3. simulation parameters
    par.simN = 10000
    par.M_ini = 1.5
    
    # 4. grid for end-of-period savings
    par.num_a = 100
    par.grid_a = nonlinspace(0 + 1e-8,par.M,par.num_a,1.1)

    # 5. dimension of value function space
    par.dim = [par.num_a,par.T]
    
    return par

def EGM_loop (sol,t,par):

    # 1. construct interpolator
    interp = interpolate.interp1d(sol.M[:, t+1],sol.C[:, t+1], bounds_error=False, fill_value = "extrapolate")
    
    # 2. loop over post-decision state
    for i_a,a in enumerate(par.grid_a): # Loop over end-of-period assets

        # FILL IN. Hint: Use the EGM step (see Bertel's or my slides)
        # 1. Find m_next given a
        # 2. Find c_next using interpolation of next period solution
        # 3. Find expected marginal utility of next period consumption
        # 4. Find optimal consumption using inverted Euler
        # 5. Find endogenous cash on hand (m)

        ### SOLUTION ###
        m_next = par.R * a + par.eps
        c_next = interp(m_next)
        EMU_next = np.sum(par.eps_w*marg_util(c_next,par))
        c_now = inv_marg_util(par.R * par.beta * EMU_next, par)
        ### SOLUTION ###

        # Index 0 is used for the corner solution, so start at index 1
        sol.C[i_a+1,t]= c_now
        sol.M[i_a+1,t]= c_now + a

    return sol

def EGM_vectorized (sol,t,par):

    interp = interpolate.interp1d(sol.M[:,t+1],sol.C[:,t+1], bounds_error=False, fill_value = "extrapolate") # Interpolation function
    
    # FILL IN. Hints:
    # - Look at the exercise_2.EGM_loop function and follow the EGM step procedure
    # - Look at the exercise_1.euler_error_func function and follow the vectorization syntax

    ### SOLUTION ###
    m_next = par.R*par.grid_a[:, None] + par.eps[None, :]
    c_next = interp(m_next)
    EU_next = np.sum(par.eps_w[None, :] * marg_util(c_next,par),axis=1)
    c_now = inv_marg_util(par.beta * par.R * EU_next, par)
    sol.C[1:,t] = c_now
    sol.M[1:,t] = c_now + par.grid_a
    ### SOLUTION ###

    return sol

def solve_EGM(par, vector = False):

    # 1. initialize solution class
    class sol: pass
    sol.C = np.zeros((par.num_a+1, par.T)) + np.nan
    sol.M = np.zeros((par.num_a+1, par.T)) + np.nan

    # 2. last period, consume everything
    sol.M[:, par.T-1] = nonlinspace(0, par.M, par.num_a+1, 1.1)
    sol.C[:, par.T-1]= sol.M[:, par.T-1].copy()

    # 3. loop over periods
    for t in reversed(range(0, par.T-1)):  # T-2, T-1, ..., 0
        if vector == True:
            sol = EGM_vectorized(sol, t, par)
        else:
            sol = EGM_loop(sol, t, par)

        # add zero consumption to account for borrowing constraint
        sol.M[0,t] = 0
        sol.C[0,t] = 0

    return sol

def simulate (par,sol):
    
    # 1. initialize
    class sim: pass
    sim.M = par.M_ini*np.ones((par.simN, par.T))
    sim.C = np.zeros((par.simN, par.T)) + np.nan

    # 2. simulate 
    np.random.seed(2026)
    for t in range(par.T):

        # a. construct interpolator
        interp = interpolate.interp1d(sol.M[:, t],sol.C[:, t], bounds_error=False, fill_value = "extrapolate") 
        
        # b. find consumption given state
        sim.C[:, t] = interp(sim.M[:, t])
    
        # c. state transition
        if t<par.T-1:

            # i. draw random numbers
            logY = np.random.normal(par.mu,par.sigma,par.simN)
            Y = np.exp(logY)

            # ii. compute savings
            A = sim.M[:, t]-sim.C[:, t]
        
            # iii. state transition
            sim.M[:, t+1] = par.R*A + Y
            
    return sim