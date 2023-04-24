# Import package
import numpy as np
import tools


def EGM (sol,z_plus, t,par):
    """ EGM step"""
    # Find expected value and expected marginal utility given discrete choice 
    if z_plus == 1:     #Retired
        w_raw, avg_marg_u_plus = retired(sol,z_plus,t,par)
    else:               #Working
        w_raw, avg_marg_u_plus = working(sol,z_plus,t,par)


    # Find raw consumption, cash-on-hand and value using standard EGM using the expectations found above
    c_raw = inv_marg_util(par.beta*par.R*avg_marg_u_plus,par)
    m_raw = c_raw + par.grid_a
    v_raw = util(c_raw,z_plus,par) + par.beta * w_raw

    # UPPER ENVELOPE: Clean the consumption and value functions for sub-optimal choices

    # Reorder raw grid to make it ascending in M
    # M_raw is not ascending in M by default due to the Euler having multiple solutions - check slides/paper for details
    m = sorted(m_raw)  # alternatively, choose a common grid exogeneously. This, however, creates many points around the kink
    I = m_raw 
    c = [x for _,x in sorted(zip(I,c_raw))]  # sort c according to m 
    v = [x for _,x in sorted(zip(I,v_raw))] # sort v according to m

    #If retired: No Upper-envelope
    if z_plus == 1:
        return m,c,v

    # Loop through the endogenous grid
    for i in range(np.size(m_raw)-2):
        # Create slope of consumption function between points for interpolation purposes
        m_low = m_raw[i] 
        m_high = m_raw[i+1]
        c_slope = (c_raw[i+1]-c_raw[i])/(m_high-m_low)

        # Loop through the sorted grid
        for j in range(len(m)):
            # If the point is between m_low and m_high, compute alternative guesses on consumption and value
            if  m[j]>=m_low and m[j]<=m_high:

                # Interpolate consumption and value at the point
                c_guess = c_raw[i]+c_slope*(m[j]-m_low)
                v_guess = value_of_choice_worker(m[j],c_guess,t,sol,par)
                    
                # If new guess is better than previous guess, replace old guess with new guess
                # This is where the "zig-zag"-region is cleaned
                if v_guess >v[j]:
                    v[j]=v_guess
                    c[j]=c_guess

    return m,c,v

def retired(sol,z_plus,t,par):
    """ Find expected value and expected marginal utility given retirement """
    #  prepare
    a = par.grid_a
    # Next period states - no income when retired
    m_plus = par.R*a

    #value
    w_raw=tools.interp_linear_1d(sol.m[t+1,z_plus,:],sol.v[t+1,z_plus,:], m_plus)
            
    # Consumption
    c_plus = tools.interp_linear_1d(sol.m[t+1,z_plus,:],sol.c[t+1,z_plus,:], m_plus)
            
    #Marginal utility
    marg_u_plus = marg_util(c_plus,par)

    #Expected marginal utility
    avg_marg_u_plus = marg_u_plus

    return w_raw, avg_marg_u_plus


def working(sol,z_plus,t,par):
    """ Find expected value and expected marginal utility given working """
    # Prepare - use tile/repeat to get all combinations of shocks and states
    xi = np.tile(par.xi,par.Na)
    a = np.repeat(par.grid_a,par.Nxi) 
    w = np.tile(par.xi_w,(par.Na,1))

    # Next period states
    m_plus = par.R*a+par.W*xi

    # Prepare for choice specific value, consumption and marginal utility
    shape = (2,m_plus.size)
    v_plus = np.nan+np.zeros(shape)
    c_plus = np.nan+np.zeros(shape)
    marg_u_plus = np.nan+np.zeros(shape)

    for i in range(2): #Range over working and not working next period
        # Choice specific value
        v_plus[i,:]=tools.interp_linear_1d(sol.m[t+1,i,:],sol.v[t+1,i,:], m_plus)
            
        #Choice specific consumption
        c_plus[i,:] = tools.interp_linear_1d(sol.m[t+1,i,:],sol.c[t+1,i,:], m_plus)
            
        # Choice specific Marginal utility
        marg_u_plus[i,:] = marg_util(c_plus[i,:] ,par) 

    # Expected value and
    ## Expectation over future taste shocks
    V_plus, prob = logsum(v_plus[0],v_plus[1],par.sigma_eta)
    ## Expectation over future income shocks
    w_raw = w*np.reshape(V_plus,(par.Na,par.Nxi))
    w_raw = np.sum(w_raw,1)


    #Expected  average marg. utility
    ## Expectation over future taste shocks
    marg_u_plus = prob[0,:]*marg_u_plus[0] + prob[1,:]*marg_u_plus[1]  
    ## Expectation over future income shocks
    avg_marg_u_plus = w*np.reshape(marg_u_plus,(par.Na,par.Nxi))
    avg_marg_u_plus = np.sum(avg_marg_u_plus ,1)

    return w_raw, avg_marg_u_plus


def value_of_choice_worker(m,c,t,sol,par):
    """ Value of choice given current state and consumption choice given wirker """
    L = 0
    # Prepare - use tile/repeat to get all combinations of shocks and states
    xi_w_mat = np.tile(par.xi_w,(c.size,1))
    xi_mat = np.tile(par.xi,(c.size))
    # Next period ressources
    a = np.repeat(m-c,(par.xi.size))
    m_plus = par.R * a + par.W*xi_mat

    # Next-period value
    ## Expectation over future taste shocks
    v_plus0 = tools.interp_linear_1d(sol.m[t+1,0,par.N_bottom:],sol.v[t+1,0,par.N_bottom:], m_plus)
    v_plus1 = tools.interp_linear_1d(sol.m[t+1,1,par.N_bottom:],sol.v[t+1,1,par.N_bottom:], m_plus)
    V_plus, _ = logsum(v_plus0,v_plus1,par.sigma_eta)
    ## Expectation over future income shocks
    V_plus = np.reshape(V_plus,(c.size,par.xi_w.size))
    V_plus = np.sum(xi_w_mat*V_plus,1)

    # This period value
    v = util(c,L,par)+par.beta*V_plus
    return v


def value_of_choice_retired(m,c,t,sol,par):
    """ Value of choice given current state and consumption choice given retired """
    #prepare
    L = 1
    z_plus = 1
    a = m-c
    # Next period states
    m_plus = par.R*a

    #value
    V_plus=tools.interp_linear_1d(sol.m[t+1,z_plus,:],sol.v[t+1,z_plus,:], m_plus)

    # This period value
    v = util(c,L,par)+par.beta*V_plus
    return v


    #FUNCTIONS
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
