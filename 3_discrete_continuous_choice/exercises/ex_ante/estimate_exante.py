# Import package
import numpy as np
import scipy.optimize as optimize
import tools
import model_exante as model

def maximum_likelihood(model, est_par, theta0, data,do_stderr):
    
    # Check the parameters
    assert (len(est_par)==len(theta0)), 'Number of parameters and initial values do not match'
    
    #Estimation
    # Fill in

    return res

def log_likelihood(theta, model, est_par, data):
    
    #Update parameters
    par = model.par
    sol = model.sol

    par = updatepar(par,est_par,theta)

    # Solve the model
    model.create_grids()
    model.solve()

    # Predict consumption
    t = data.t
    c_predict = tools.interp_linear_1d(sol.m[t,:],sol.c[t,:],data.m)
    C_predict = c_predict*data.P        #Renormalize

    # Calculate errors
    error = data.logC -np.log(C_predict)

    # Calculate log-likelihood
    # Fill in
    
    return np.mean(log_lik_vec) 

def updatepar(par, parnames, parvals):

    for i,parval in enumerate(parvals):
        parname = parnames[i]
        setattr(par,parname,parval) # It gives the attibute parname the new value parval, within the par class
    return par

def calc_moments(par,data):
    agegrid = np.arange(par.moments_minage,par.moments_maxage+1)-par.age_min+1 # define the cell which correspond to the age we want the mean for. e.g. age 40-55 --> agegrid: 16-31
    return np.mean(data.A[agegrid,:],1)

def method_simulated_moments(model,est_par,theta0,data):


    # Check the parameters
    assert (len(est_par)==len(theta0)), 'Number of parameters and initial values do not match'
    
    # Calculate data moments
    data.moments = calc_moments(model.par,data)

    # Estimate
    # Fill in
    res = optimize.minimize(obj_fun,theta0, method='BFGS')

    return res

def sum_squared_diff_moments(theta,model,est_par,data):

    par = model.par
    #Update parameters
    par = updatepar(par,est_par,theta)

    # Solve the model
    model.create_grids()
    model.solve()

    # Simulate the momemnts
    moments = np.nan + np.zeros((data.moments.size,par.moments_numsim))
    for s in range(par.moments_numsim):

        # Simulate
        model.simulate()

        #Calculate moments
        moments[:,s] = calc_moments(par,model.sim)

    # Mean of moments         
    moments = np.mean(moments,1)

    # Objective function
    if hasattr(par, 'weight_mat'):
        weight_mat_inv = np.linalg.inv(par.weight_mat)  
    else:
        weight_mat_inv = np.eye(moments.size)   # The identity matrix and I^-1=I
    
    diff = (moments-data.moments).reshape(moments.size,1)

    return (# Fill in).ravel()
