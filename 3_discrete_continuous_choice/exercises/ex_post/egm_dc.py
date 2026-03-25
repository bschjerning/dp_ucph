import numpy as np
import tools

def EGM(sol, z_plus, t,par):

    # 1. find expected value and expected marginal utility given discrete choice 
    if z_plus == 1: # retired
        EV_raw, EMU_plus = retired(sol,z_plus,t,par)
    else: # working
        EV_raw, EMU_plus = working(sol,z_plus,t,par)

    # 2. find raw consumption, cash-on-hand and value
    
    # FILL IN. Delete None. Hint: Use standard EGM using the expectations found above
    c_raw = None
    m_raw = None
    v_raw = None

    ### SOLUTION ###
    c_raw = inv_marg_util(par.beta * par.R * EMU_plus, par)
    m_raw = c_raw + par.grid_a
    v_raw = util(c_raw, z_plus, par) + par.beta * EV_raw
    ### SOLUTION ###

    # 3. reorder raw grid to make it ascending in M, can be not ascending in m, since Euler can have multiple solutins
    m = sorted(m_raw)
    I = m_raw
    c = [x for _,x in sorted(zip(I,c_raw))]  # sort c according to m 
    v = [x for _,x in sorted(zip(I,v_raw))] # sort v according to m

    # 4. if retired -> no Upper-envelope
    if z_plus == 1:
        return m,c,v

    # 5. loop through the endogenous grid
    for i in range(np.size(m_raw)-2):

        # a. create slope of consumption function between points for interpolation purposes
        m_low = m_raw[i] 
        m_high = m_raw[i+1]
        c_slope = (c_raw[i+1]-c_raw[i])/(m_high-m_low)

        # b. loop through the sorted grid
        for j in range(len(m)):

            # if the point is between m_low and m_high, compute alternative guesses on consumption and value
            if  m_low<=m[j] and m[j]<=m_high:

                # interpolate consumption and value at the point
                c_guess = c_raw[i] + c_slope * (m[j] - m_low)
                v_guess = value_of_choice_worker(m[j], c_guess, t, sol, par)
                v_guess = v_guess.item() # convert to scalar to avoid problems with lists of numbers and numpy arrays
                    
                # If new guess is better than previous guess, replace old guess with new guess
                # This is where the "zig-zag"-region is cleaned
                if v_guess >v[j]:
                    v[j]=v_guess
                    c[j]=c_guess

    return m,c,v

def retired(sol, z_plus, t, par):
    """ Find expected value and expected marginal utility given retirement """

    a = par.grid_a

    # 1. state transition - no income when retired
    m_plus = par.R*a

    # 2. interpolate choice-specific value
    EV_raw = tools.interp_linear_1d(sol.m[t+1, z_plus, :], sol.v[t+1, z_plus, :], m_plus)
            
    # 3. interpolate consumption
    c_plus = tools.interp_linear_1d(sol.m[t+1, z_plus, :], sol.c[t+1, z_plus, :], m_plus)
            
    # 4. compute marginal utility
    MU_plus = marg_util(c_plus,par)

    # 5. compute expected marginal utility: no shocks when retired
    EMU_plus = MU_plus

    return EV_raw, EMU_plus

def working(sol,z_plus,t,par):

    # 1. prepare - use tile and repeat to get all combinations of shocks and states
    xi = np.tile(par.xi,par.Na)
    a = np.repeat(par.grid_a,par.Nxi) 
    w = np.tile(par.xi_w,(par.Na,1))

    # 2. state transition
    m_plus = par.R*a+par.W*xi

    # 3. allocate for choice specific value, consumption and marginal utility
    v_plus = np.zeros((2, m_plus.size)) + np.nan
    c_plus = np.zeros((2, m_plus.size)) + np.nan
    MU_plus = np.zeros((2, m_plus.size)) + np.nan

    # 4. interpolate choice-specific value and consumption for next period
    for i in range(2):

        # a. choice specific value
        v_plus[i, :]=tools.interp_linear_1d(sol.m[t+1, i, :], sol.v[t+1, i, :], m_plus)
            
        # b. choice specific consumption
        c_plus[i, :] = tools.interp_linear_1d(sol.m[t+1, i, :], sol.c[t+1, i, :], m_plus)
            
        # c. choice specific marginal utility
        MU_plus[i, :] = marg_util(c_plus[i, :], par) 

    # 5. compute expected value function for next period
    V_plus, prob = logsum(v_plus[0], v_plus[1], par.sigma_eta)
    EV_raw = w * np.reshape(V_plus, (par.Na, par.Nxi))
    EV_raw = np.sum(EV_raw, axis = 1)

    # 6. compute expected marginal utility for next period
    MU_plus = prob[0, :]*MU_plus[0] + prob[1, :]*MU_plus[1]  
    EMU_plus = w*np.reshape(MU_plus, (par.Na, par.Nxi))
    EMU_plus = np.sum(EMU_plus, axis = 1)

    return EV_raw, EMU_plus

def value_of_choice_worker(m,c,t,sol,par):

    # 1. set retirement state to not retired
    L = 0

    # 2. prepare - use tile/repeat to get all combinations of shocks and states
    xi_w_mat = np.tile(par.xi_w, (c.size, 1))
    xi_mat = np.tile(par.xi, (c.size))

    # 3. state transition
    a = np.repeat(m - c, (par.xi.size))
    m_plus = par.R * a + par.W * xi_mat

    # 4. next period choice-specific values
    v_plus0 = tools.interp_linear_1d(sol.m[t+1, 0, par.N_bottom:], sol.v[t+1, 0, par.N_bottom:], m_plus)
    v_plus1 = tools.interp_linear_1d(sol.m[t+1, 1, par.N_bottom:], sol.v[t+1 ,1, par.N_bottom:], m_plus)
    V_plus, _ = logsum(v_plus0, v_plus1, par.sigma_eta)

    # 5. compute expected next period value
    V_plus = np.reshape(V_plus, (c.size, par.xi_w.size))
    V_plus = np.sum(xi_w_mat * V_plus, 1)

    # 6. compute value in this period
    v = util(c, L, par) + par.beta * V_plus

    return v

def value_of_choice_retired(m,c,t,sol,par):

    # 1. prepare
    L = 1
    z_plus = 1
    a = m - c

    # 2. state transition
    m_plus = par.R*a

    # 3. interpolate value function next period
    V_plus = tools.interp_linear_1d(sol.m[t+1, z_plus, :], sol.v[t+1, z_plus,:], m_plus)

    # 4. compute value function this period
    v = util(c, L, par) + par.beta * V_plus
    
    return v

#############
# FUNCTIONS #
#############
 
def util(c,L,par):
    """ Utility function"""
    return ((c**(1.0-par.rho))/(1.0-par.rho)-par.alpha*(1-L))

def marg_util(c,par):
    """ Marginal utility function"""
    return c**(-par.rho)

def inv_marg_util(u,par):
    """ Inverse of marginal utility function"""
    return u**(-1/par.rho)

def logsum(v1,v2,sigma):
    """ Numerically robust log-sum calculation"""

    # setup
    V = np.array([v1, v2])

    # Maximum over the discrete choices
    mxm = V.max(0)

    # check the value of sigma
    if abs(sigma) > 1e-10:

        # numerically robust log-sum
        log_sum = mxm + sigma*(np.log(np.sum(np.exp((V - mxm) / sigma),axis=0)))
        
        # d. numerically robust probability
        prob = np.exp((V- log_sum) / sigma)    

    else: # No smoothing --> max-operator
        id = V.argmax(0)    #Index of maximum
        log_sum = mxm
        prob = np.zeros((v1.size*2))
        I = np.cumsum(np.ones((v1.size,1)))+id*(v1.size)-1
        I = I.astype(int)  # change type to integer
        prob[I] = 1
        prob = np.reshape(prob,(2,v1.size),'A')

    return log_sum,prob
