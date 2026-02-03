# import packages used
import numpy as np

def solve_VFI(par):

    # 1. grid for policy
    Cstar = np.zeros([par.W+1])
    
    # 2. parameters for VFI
    max_iter = par.max_iter # maximum number of iterations
    tol = par.tol #convergence tol. level
    
    delta = 1000 # difference between V_next and V_now
    it = 0  #iteration counter 
    V_now = np.zeros([par.W+1]) #arbitrary starting values
    
    # 3. iterate
    while (it <= max_iter and tol < delta):
        it = it+1
        V_next = V_now.copy() # make copy to avoid in-place

        # loop over states
        for w_i in range(par.W+1):
            
            # FILL IN. 

            ### SOLUTION ###
            c = np.arange(w_i+1)
            w_next = w_i - c
            V_search = np.sqrt(c) + par.beta*V_now[w_next]
            V_next[w_i] = np.amax(V_search)
            Cstar[w_i] = c[np.argmax(V_search)]
            ### SOLUTION ###
            
        delta = np.amax(np.abs(V_now - V_next))

        V_now = V_next.copy()
    
    # 4. setup class for saving results
    class sol: pass
    sol.C = Cstar
    sol.V = V_now
    sol.it = it

    return sol