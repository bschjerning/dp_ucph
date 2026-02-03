# import packages used
import numpy as np

def solve_consumption_uncertainty(par):

    # 1. initialize solution class and allocate memory
    class sol: pass
    sol.V = np.zeros([par.num_W,par.T]) + np.nan
    sol.C = np.zeros([par.num_W,par.T]) + np.nan
    sol.grid_W = np.zeros([par.num_W,par.T]) + np.nan
    
    # 2. define consumption "guess" grid as a share of available resources
    grid_C = np.linspace(0.0,1.0,par.num_C)
    
    # 3. Backwards Induction: Loop over periods
    for t in reversed(range(par.T)):  #from period T-1, until period 0, backwards

        # let grid for W depend on the maximum attainable cake size in period
        W_max = max(par.eps)*t+par.W # maxmium cake size is expanding over t
        grid_W = np.linspace(0,W_max,par.num_W)
        sol.grid_W[:,t] = grid_W 
    
        for iw,w in enumerate(grid_W):
            c = grid_C*w
            w_c = w - c
            EV_next = 0
        
            if t<par.T-1: # no EV_next in last period
                pass # delete this, just there to make import work with no code in loop
                
                # FILL IN.
                # Hint: 1) Loop through shock probability and values, e.g. by using the zip function
                #       2) Interpolate value function for the new state given each shock
                #       3) Add probability-weighted contribution to expectation

                ### SOLUTION ###
                for pi_i, eps_i in zip(par.pi, par.eps):
                    EV_next += pi_i * np.interp(w_c + eps_i, sol.grid_W[:, t+1], sol.V[:, t+1])
                ### SOLUTION ###
                
            V_guess = np.sqrt(c)+par.beta*EV_next
            index = np.argmax(V_guess)
            sol.C[iw,t] = c[index]
            sol.V[iw,t] = np.amax(V_guess)
        
    return sol