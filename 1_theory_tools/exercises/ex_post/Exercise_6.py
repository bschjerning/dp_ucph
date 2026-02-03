# import packages used
import numpy as np

def util(c,par):
    return (c**(1.0-par.rho))/(1.0-par.rho)

def solve_consumption_deaton(par):

    # 1. initialize solution class
    class sol: pass

    # 2. allocate memory
    sol.V = np.zeros([par.num_W,par.T]) + np.nan
    sol.C = np.zeros([par.num_W,par.T]) + np.nan
    sol.grid_W = np.zeros([par.num_W,par.T]) + np.nan
    
    # 3. consumption grid as a share of available resources
    grid_C = np.linspace(0.0,1.0,par.num_C)
    
    # 4. Loop over periods
    for t in range(par.T-1, -1, -1):  #from period T-1, until period 0, backwards 

        # setup grid for W in period t
        W_max = max(par.eps)*t+par.W
        grid_W = np.linspace(0,W_max,par.num_W) 
        sol.grid_W[:,t] = grid_W
    
        for iw,w_i in enumerate(grid_W):
            c = grid_C*w_i
            w_c = w_i - c
            EV_next = 0
        
            if t<par.T-1:
                for s in range(par.num_shocks):
                    
                    pass # delete this, just there to make import work with no code in loop
                    # FILL IN. Hint: Same procedure as in Exercise_4. With quadrature, it is as-if, we worked with discrete shocks

                    ### SOLUTION ###
                    eps_i = par.eps[s]
                    eps_w_i = par.eps_w[s] 
                    w_next_i = par.R*w_c+eps_i
                    EV_next += eps_w_i*np.interp(w_next_i, sol.grid_W[:, t+1], sol.V[:, t+1])
                    ### SOLUTION ###
                    
            V_search = util(c,par)+par.beta*EV_next
            index = np.argmax(V_search)
            sol.C[iw,t] = c[index]
            sol.V[iw,t] = np.amax(V_search)
        
    return sol