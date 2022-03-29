import numpy as np

# model solution
def c_star(w,e,par):
    return par['gamma']*(1.0-par['tau'])*w + par['gamma']*e

def l_star(w,e,par):
    return (1.0-par['gamma']) + (1.0-par['gamma'])/(1.0-par['tau'])*(e/w)
        
# objective function
def obj_fun(theta,est_par,w,mom_data,moments_fun,par):
            
    # a. update parameter struct with elements in theta
    for name,value in zip(est_par,theta):
        par[name] = value

    # b. draw random draws
    np.random.seed(893245) # note: different from the seed used for the true data
    S = 100 # number of simulation draws for each observed individual

    n = w.size # number of individuals
    w = np.tile(w,(1,S)) # stack observed wages S times
    e = par['sigma']*np.random.normal(size=n*S)

    con = c_star(w,e,par)
    lab = l_star(w,e,par)
    
    # c. calculate moments based on simulated data for the value of theta
    mom_sim = moments_fun(w,con,lab)
    
    # d. calculate objective function as squared difference
    distance = mom_data - mom_sim
    Q = distance.T@distance

    return Q