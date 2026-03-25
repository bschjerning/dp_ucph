# Import package
import numpy as np
import scipy.optimize as optimize
import tools

def maximum_likelihood(model, est_par, theta0, data):
    
    # 1. check the parameters
    assert (len(est_par)==len(theta0)), 'Number of parameters and number of initial values do not match'
    
    # 2. estimation: minimize negative log-likelihood
    obj_fun = lambda x: -log_likelihood(x,model,est_par,data)
    res = optimize.minimize(obj_fun,theta0)

    return res

def log_likelihood(theta, model, est_par, data):
    
    sol = model.sol
    
    # 1. update parameters
    par = updatepar(model.par, est_par, theta)

    # FILL IN. Delete None. Hints:
    # 1) create new grids and solve model with new parameters
    # 2) predict consumption in ratio-form in time period used for estimation
    # 3) renormalize to get consumption in non-ratio form
    # 4) compute measurement errors for logged consumption
    # 5) compute log-likelihood when error follows a normal distribution

    log_lik = None

    ### SOLUTION ###
    model.create_grids()
    model.solve()
    c_predict = tools.interp_linear_1d(sol.m[data.t, :], sol.c[data.t, :], data.m)
    C_predict = c_predict*data.P
    error = data.logC -np.log(C_predict)
    log_lik_vec = - 0.5*np.log(2*np.pi*par.sigma_eta**2)
    log_lik_vec += (- (error**2)/(2*par.sigma_eta**2) )
    log_lik = np.mean(log_lik_vec) 
    ### SOLUTION ###
    
    return log_lik

def updatepar(par, parnames, parvals):

    for i,parval in enumerate(parvals):
        parname = parnames[i]
        setattr(par,parname,parval) # it gives the attibute parname the new value parval, within the par class

    return par

def calc_moments(par,data):
    
    # define the cell which correspond to the age we want the mean for. e.g. age 40-55 --> agegrid: 16-31
    agegrid = np.arange(par.moments_minage, par.moments_maxage+1)-par.age_min+1 

    return np.mean(data.A[agegrid, :], axis=1)

def method_simulated_moments(model,est_par,theta0,data):

    # 1. check the parameters
    assert (len(est_par)==len(theta0)), 'Number of parameters and initial values do not match'
    
    # 2. calculate data moments
    data.moments = calc_moments(model.par,data)

    # 3. estimate
    obj_fun = lambda x: sum_squared_diff_moments(x,model,est_par,data)
    res = optimize.minimize(obj_fun,theta0, method='BFGS')

    return res

def sum_squared_diff_moments(theta,model,est_par,data):

    # 1. update parameters
    par = model.par
    par = updatepar(par, est_par, theta)

    # FILL IN. Delete None. Hints:
    # 1) create new grids and solve model with new parameters
    # 2) simulate the moment(s) (in loop if moments across several simulations)
    # 3) compute mean of moments across potentially several simulations in loop
    # 4) compute objective function using inverse weighting matrix

    diff_moments = None

    ### SOLUTION ###
    model.create_grids()
    model.solve()
    moments = np.zeros((data.moments.size, par.moments_numsim)) + np.nan
    for s in range(par.moments_numsim):
        # a. simulate
        model.simulate()
        # b. calculate moments
        moments[:,s] = calc_moments(par,model.sim)
    moments = np.mean(moments, axis=1)
    if hasattr(par, 'weight_mat'):
        weight_mat_inv = np.linalg.inv(par.weight_mat)  
    else:
        weight_mat_inv = np.eye(moments.size)   # identity matrix and I^-1=I
    diff = (moments-data.moments).reshape(moments.size,1)
    diff_moments = (diff.T @ weight_mat_inv @ diff)
    ### SOLUTION ###

    return diff_moments
