# import packages used
import numpy as np

def solve_VFI(par):
    Cstar = np.zeros([par.W+1])
    
    # Parameters for VFI
    max_iter = par.max_iter   # maximum number of iterations
    delta = 1000 #difference between V_next and V_now
    tol = par.tol #convergence tol. level
    it = 0  #iteration counter 
    V_now = np.zeros([par.W+1]) #arbitrary starting values
    
    while (max_iter>= it and tol<delta):
        it = it+1
        V_next = V_now.copy()
        for w in range(par.W+1):
            #Fill in
            
            
            
            
        delta = np.amax(np.abs(V_now - V_next))
    
    class sol: pass
    sol.C = Cstar
    sol.V = V_now
    sol.it = it

    return sol