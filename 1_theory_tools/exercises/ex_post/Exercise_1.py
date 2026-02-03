# import packages used
import numpy as np

def solve_backwards(beta,W,T):

    # 1. allocate memory
    Cstar_bi = np.zeros([W+1,T]) + np.nan 
    V_bi = np.zeros([W+1,T]) + np.nan
    
    # 2. solve static problem in last period
    Cstar_bi[:,T-1] = np.arange(W+1) 
    V_bi[:,T-1] = np.sqrt(Cstar_bi[:,T-1])

    # 3. solve
    # Loop over periods
    for t in reversed(range(T-1)):  #from period T-2, until period 0, backwards  
        
        #loop over states
        for w_i in range(W+1):
            
            c = np.arange(w_i+1)
            
            # FILL IN. Hint: Use your code from the notebook

            ### SOLUTION ###
            w_next = w_i - c
            V_next = V_bi[w_next, t+1]
            V_guess = np.sqrt(c)+beta*V_next
            ### SOLUTION ###
            
            V_bi[w_i,t] = np.amax(V_guess)
            Cstar_bi[w_i,t] = np.argmax(V_guess)

    return Cstar_bi, V_bi