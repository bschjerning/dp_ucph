import numpy as np
import tools

def EGM (sol,z_plus,p, t,par): 

    
    return c,v


def retired(sol,z_plus,p, t,par):
    
    return w_raw, avg_marg_u_plus

def working(sol,z_plus,p, t,par):
    

    return w_raw, avg_marg_u_plus

def upper_envelope(t,z_plus,c_raw,m_raw,w_raw,par):
    

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
