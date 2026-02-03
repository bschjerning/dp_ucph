# import packages
import numpy as np
import scipy.optimize as optimize

def solve_consumption_grid_search(par):

    # 1. initialize solution class
    class sol: pass

    # 2. allocate memory
    sol.C = np.zeros(par.num_W) + np.nan
    sol.V = np.zeros(par.num_W) # dont fill in with nans here as we use specific values
    
    # 3. consumption grid as a share of available resources, like grid over saving rates
    grid_C = np.linspace(0.0,1.0,par.num_C) 

    # 4. parameters for VFI
    delta = 1000 # difference between V_next and V_now (arbitrary)
    it = 0  # iteration counter 
    
    # 5. iterate
    while (it <= par.max_iter and delta > par.tol):
        it = it+1
        V_next = sol.V.copy()

        # loop over states
        for iw,w_i in enumerate(par.grid_W): # need index for grids now too

            # FILL IN
            # Hint: 1) For each w create a consumption grid, c, using grid_C.
            #       2) Use c to calculate V_search using interpolation
            #       3) In order to interpolate use: np.interp
            #       4) Proceed as in Exercise_2.py, only new thing is interpolation

            ### SOLUTION ###
            c = grid_C * w_i
            w_next = w_i - c
            V_search = np.sqrt(c) + par.beta * np.interp(w_next, par.grid_W, sol.V)
            V_next[iw] = np.amax(V_search)
            sol.C[iw] = c[np.argmax(V_search)]
            ### SOLUTION ###
      
        delta = np.amax(np.abs(sol.V - V_next))

        # update solution
        sol.V = V_next.copy()
    
    return sol