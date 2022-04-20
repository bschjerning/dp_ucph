# import packages used
import numpy as np
from scipy import interpolate

def util(c,L,par):
    return (c**(1.0-par.rho))/(1.0-par.rho)-par.lambdaa*L

def setup():
    class par: pass

    par.T = 10

    # Model parameters
    par.beta = 0.98
    par.rho = 0.5
    par.lambdaa = 1.3
    par.R = 1.0/par.beta
    par.W = 1.0
    par.sigma = 0.5
    par.L = 2

    # Cash-on-hand and consumption parameters
    par.num_M = 100
    par.M_max = 10
    par.num_C = 100

    # Simulation parameters
    par.simN = 10000
    par.M_ini = 1.5

    #Grid of cash-on-hand and consumption
    par.grid_M = nonlinspace(1.0e-6,par.M_max,par.num_M,1.1) # same as np.linspace just with unequal spacing
    par.grid_C = np.linspace(1.0e-6,1.0,par.num_C)
    
    # Dimension of value function space
    par.dim = [par.L,par.num_M,par.T]
    
    return par


def solve_DC(par):
     # initialize solution class
    class sol: pass
    sol.V = np.zeros(par.dim) 
    sol.C = np.zeros(par.dim)
    
    # Loop over periods
    for t in range(par.T-1, -1, -1):  #from last period until period 0, backwards 
        
        EV_next = 0
        
        for L in range(par.L):
        
            for im,m in enumerate(par.grid_M):
                c = par.grid_C*m
                
                if t<par.T-1:
                    interp0 = interpolate.interp1d(par.grid_M,sol.V[0,:,t+1], bounds_error=False, fill_value = "extrapolate")
                    interp1 = interpolate.interp1d(par.grid_M,sol.V[1,:,t+1], bounds_error=False, fill_value = "extrapolate")
                  
                    m_next = par.R*(m-c)+par.W*L
            
                    V0 = interp0(m_next)
                    V1 = interp1(m_next)
            
                    # Compute the log-sum
                    maxM = np.maximum(V0,V1)
                    EV_next = maxM +par.sigma*np.log(np.exp((V0-maxM)/par.sigma)+np.exp((V1-maxM)/par.sigma))    
                
                V_guess = util(c,L,par)+par.beta*EV_next
                index = np.argmax(V_guess)
                sol.C[L,im,t] = c[index]
                sol.V[L,im,t] = np.amax(V_guess) 
                
    return sol

def simulate (par,sol):
    
    # Initialize
    class sim: pass
    shape = (par.simN,par.T)
    sim.M = par.M_ini*np.ones(shape)
    sim.C = np.nan +np.zeros(shape)
    sim.L = np.nan +np.zeros(shape)
    np.random.seed(2022)

    # Random numbers
    eps = np.random.rand(par.simN,par.T) # uniform distirbuted

    # Simulate 
    for t in range(par.T):

        # Values of discrete choice
        interp0 = interpolate.interp1d(par.grid_M,sol.V[0,:,t], bounds_error=False, fill_value = "extrapolate")
        interp1 = interpolate.interp1d(par.grid_M,sol.V[1,:,t], bounds_error=False, fill_value = "extrapolate")
        
        # Interpreted values for value function
        V0 = interp0(sim.M[:,t]) 
        V1 = interp1(sim.M[:,t])
        
        # Work choice
        prob = 1/(1+np.exp((V0-V1)/par.sigma))  # probabilty of working for each person given state  
        I = eps[:,t] <= prob # Indicator function for working

        # Consumption of discrete choice
        interpc0 = interpolate.interp1d(par.grid_M,sol.C[0,:,t], bounds_error=False, fill_value = "extrapolate")
        interpc1 = interpolate.interp1d(par.grid_M,sol.C[1,:,t], bounds_error=False, fill_value = "extrapolate")
        
        # Interpreted values for consumption
        C0 = interpc0(sim.M[:,t]) 
        C1 = interpc1(sim.M[:,t]) 

        sim.C[I,t] = C0[I]   #Consumption for people working
        sim.C[~I,t] = C1[~I] # Consumption for people not working
    
        # Labour choice
        sim.L[~I,t] = 0
        sim.L[I,t] = 1

        # Next period
        if t<par.T-1:  # if not last period
            A = sim.M[:,t]-sim.C[:,t]
            sim.M[:,t+1] = par.R*A + par.W*sim.L[:,t] # The state in the following period
            
     
    return sim


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