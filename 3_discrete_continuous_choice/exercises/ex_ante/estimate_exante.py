# Import package
import numpy as np
import scipy.optimize as optimize
import tools
import model_exante as model

def maximum_likelihood(par, est_par, theta0, data,do_stderr):
    
    # Check the parameters
    assert (len(est_par)==len(theta0)), 'Number of parameters and initial values do not match'
    
    #Estimation
    obj_fun = lambda x: -log_likelihood(x,est_par,par,data)
    res = optimize.minimize()

    return res

def log_likelihood(theta, est_par, par, data):
    
    #Update parameters
    par = updatepar(par,est_par,theta)

    # Solve the model
    par = model.create_grids(par)
    sol = model.solve(par)

    # Predict consumption

    # Calculate errors


    # Calculate log-likelihood
    
    return f

def updatepar(par, parnames, parvals):

    for i,parval in enumerate(parvals):
        parname = parnames[i]
        setattr(par,parname,parval) # It gives the attibute parname the new value parval, within the par class
    return par

def calc_moments(par,data):
    agegrid = np.arange(par.moments_minage,par.moments_maxage+1)-par.age_min+1
    return np.mean(data.A[agegrid,:],1)


def method_simulated_moments(par,est_par,theta0,data):

    # Check the parameters
    assert (len(est_par)==len(theta0)), 'Number of parameters and initial values do not match'
    
    # Calculate data moments
    data.moments = calc_moments(par,data)

    # Estimate
    obj_fun = lambda x: sum_squared_diff_moments(x,est_par,par,data)
    res = optimize.minimize()

    return res


def sum_squared_diff_moments(theta,est_par,par,data):

    #Update parameters
    par = updatepar(par,est_par,theta)

    # Solve the model
    par = model.create_grids(par)
    sol = model.solve(par)

    # Simulate the momemnts
    moments = np.nan + np.zeros((data.moments.size,par.moments_numsim))
    for s in range(par.moments_numsim):
        print(s)

    # Mean of moments         

    # Objective function
    

    return obj
