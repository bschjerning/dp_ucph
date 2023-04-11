# Import package
import numpy as np
import scipy.optimize as optimize
import tools
import model_exante as model

def maximum_likelihood(model, est_par, theta0, data):
    
    # Check the parameters
    assert (len(est_par)==len(theta0)), 'Number of parameters and initial values do not match'
    
    #Estimation
    obj_fun = lambda x: -log_likelihood(x,model,est_par,data)
    res = optimize.minimize(obj_fun,theta0)

    return res

def log_likelihood(theta, model, est_par, data):
    
    #Unpack
    par = model.par
    sol = model.sol
    
    #Update parameters
    par = updatepar(par,est_par,theta)

    # Fill in
    # Hint: Remember to update the grids when parameters change
    #       Renormalize consumption to fit to data
    #       Error in predicted consumption is normally distributed (See Bertel's slides)
    
    # Predict consumption

    # Calculate errors

    # Calculate log-likelihood
    
    # return loglikelihood

def updatepar(par, parnames, parvals):
    ''' Update parameter values in par of parameters in parnames '''

    for i,parval in enumerate(parvals):
        parname = parnames[i]
        setattr(par,parname,parval) # It gives the attibute parname the new value parval, within the par class
    return par

def calc_moments(par,data):
    ''' Calculate average savings for each selected age group'''
    
    agegrid = np.arange(par.moments_minage,par.moments_maxage+1)-par.age_min+1 # define the cell which correspond to the age we want the mean for. e.g. age 40-55 --> agegrid: 16-31
    return np.mean(data.A[agegrid,:],1)

def method_simulated_moments(model,est_par,theta0,data):

    # Check the parameters
    assert (len(est_par)==len(theta0)), 'Number of parameters and initial values do not match'
    
    # Calculate data moments
    data.moments = calc_moments(model.par,data)

    # Estimate
    obj_fun = lambda x: sum_squared_diff_moments(x,model,est_par,data)
    res = optimize.minimize(obj_fun,theta0, method='BFGS')

    return res

def sum_squared_diff_moments(theta,model,est_par,data):

    #Update parameters
    par = model.par
    par = updatepar(par,est_par,theta)

    # Fill in
    # Hint: Remember to update the grids when parameters change
    #       Use an identity matrix as weighing matrix
    #       See Bertel's slides
    
    # Simulate the momemnts
    moments = np.nan + np.zeros((data.moments.size,par.moments_numsim))
    for s in range(par.moments_numsim):
        pass 

    # Mean of moments         

    # Objective function

    # return obj
