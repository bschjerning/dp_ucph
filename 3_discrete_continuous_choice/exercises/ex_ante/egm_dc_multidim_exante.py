import numpy as np
import tools

def EGM (sol,z_plus,p, t,par): 
    """ Solve model with EGM"""
   
    # Upper Envelope
    c,v = upper_envelope(t,z_plus,c_raw,m_raw,w_raw,par)
    
    return c,v


def retired(sol,z_plus,p, t,par):


    return w_raw, avg_marg_u_plus

def working(sol,z_plus,p, t,par):


    return w_raw, avg_marg_u_plus

def upper_envelope(t,z_plus,c_raw,m_raw,w_raw,par):
    """ Upper envelope and re-interpolation onto common exogenous grid of m"""
    
    # Add a point at the bottom for interpolation purposes
    c_raw = np.append(1e-6,c_raw)  
    m_raw = np.append(1e-6,m_raw) 
    a_raw = np.append(0,par.grid_a) 
    w_raw = np.append(w_raw[0],w_raw)

    # Initialize c and v   
    c = np.nan + np.zeros((par.Nm))
    v = -np.inf + np.zeros((par.Nm))
    
    # Loop through the endogenous grid
    size_m_raw = m_raw.size
    for i in range(size_m_raw-1):    

        # Get slope of consumption in m at current point for interpolation purposes
        c_now = c_raw[i]        
        m_low = m_raw[i]
        m_high = m_raw[i+1]
        c_slope = (c_raw[i+1]-c_now)/(m_high-m_low)
        # Get slope of value in a at current point for interpolation purposes
        w_now = w_raw[i]
        a_low = a_raw[i]
        a_high = a_raw[i+1]
        w_slope = (w_raw[i+1]-w_now)/(a_high-a_low)


        # Loop through the common exogenous grid grid
        for j, m_now in enumerate(par.grid_m):
            # Check whether current point is in between the raw points or above the largest raw point
            interp = (m_now >= m_low) and (m_now <= m_high) 
            extrap_above = (i == size_m_raw-1) and (m_now > m_high)
            if interp or extrap_above:
                # Interpolate Consumption
                c_guess = c_now+c_slope*(m_now-m_low)
                
                # Interpolate post-decision values
                a_guess = m_now - c_guess
                w = w_now+w_slope*(a_guess-a_low)
                
                # Value of choice
                v_guess = util(c_guess,z_plus,par)+par.beta*w
                
                # if value of choice is higher for this guess than previous value, replace
                if v_guess >v[j]:
                    v[j]=v_guess
                    c[j]=c_guess

    return c,v


# FUNCTIONS
def util(c,L,par):
    return ((c**(1.0-par.rho))/(1.0-par.rho)-par.alpha*(1-L))


def marg_util(c,par):
    return c**(-par.rho)


def inv_marg_util(u,par):
    return u**(-1/par.rho)


def logsum(v1,v2,sigma):

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

    else: # No smmothing --> max-operator
        id = V.argmax(0)    #Index of maximum
        log_sum = mxm
        prob = np.zeros((v1.size*2))
        I = np.cumsum(np.ones((v1.size,1)))+id*(v1.size)-1
        I = I.astype(int)  # change type to integer
        prob[I] = 1

        prob = np.reshape(prob,(2,v1.size),'A')

    return log_sum,prob
