#  NFXP class for structural estimation of discrete choice models.

import numpy as np
import scipy.optimize as optimize

# declare Vbar as global variable, which is normally not recommended but in this case
# makes sense since we are solving similar models all the time and can use guesses from
# earlier model solutions
global Vbar
Vbar = np.zeros(1)

def estimate(model, solver, data, theta0=[0,0], twostep=True):

    samplesize = data.shape[0]

    # 1. set ev has global variable
    global Vbar
    Vbar = np.zeros((model.n)) 
    
    # 2. find p non-parametrically by the relative frequencies of each milage change
    tabulate = data.dx1.value_counts() # Count number of observations for each dx1
    p = [tabulate[i]/sum(tabulate) if i < len(tabulate) else 0 for i in range(len(model.p))]

    # 3. estimate structual parameters by declaring them in pnames
    model.p[:] = p # Use first step estimates as starting values for p
    pnames = ['RC','c']
    
    # 4. call BHHH optimizer for loglikelihood functuon
    res = optimize.minimize(ll,theta0,
                            args = (model, solver, data, pnames),
                            method = 'trust-ncg', jac = grad, hess = hes, tol=1e-8)

    # 5. update parameters of model
    model = updatepar(model,pnames,res.x)
    
    # 6. estimate RC, c and p alltogether
    if twostep:

        # a. declare parameters to be estimated
        pnames = ['RC','c','p']

        # b. set initial guess as RC, c, p from respectively onestep estimation and non-parametric estimation
        theta0 = [model.RC, model.c] + model.p.tolist()

        # c. call BHHH optimizer
        res = optimize.minimize(ll, theta0,
                                args = (model, solver, data, pnames),
                                method = 'trust-ncg', jac = grad, hess = hes, tol = 1e-8)

        # 4. update parameters of model
        model = updatepar(model,pnames,res.x)

    # converged: "trust-ncg tends to be very conservative about convergence, and will often return status 2 even when the solution is good."
    converged = (res.status == 2 or res.status == 0)

    # 7. compute variance-covaiance matrix
    h = hes(res.x, model, solver,data, pnames) # hessian
    Avar = np.linalg.inv(h*samplesize) # variance-covariance matrix from information matrix equality

    # 8. unpack parameters and return
    theta_hat = res.x
    
    return model, res, pnames, theta_hat, Avar, converged

def ll(theta, model, solver, data, pnames, out = 1, no_guess = False): # out=1 solve optimization

    # 1. use global variable to store value function to use as starting value for next iteration
    global Vbar
    if no_guess == True: Vbar = np.zeros((model.n))
    
    # 2. unpack and convert to numpy array
    x = np.array(data.x - 1) # x is the index of the observed state: We subtract 1 because python starts counting at 0
    d = np.array(data.d) # d is the observed decision
    dx1 = np.array(data.dx1) # dx1 is observed change in x 

    # 3. update parameter values of model 
    model = updatepar(model,pnames,theta)
    model.p = np.abs(model.p) # helps BHHH which is run as unconstrained optimization
    model.create_grid() # create grid given new parameters
    Vbar0 = Vbar # use previous value function as starting value

    # 4. solve the model
    Vbar, pk, dev = solver.poly(model.bellman, V0=Vbar0, beta=model.beta, output=3)

    # 5. evaluate log-likelihood function
    lik_pr = pk[x] # get model-predicted probability of keeping given observed state    
    choice_prob = lik_pr * (d==0) + (1-lik_pr) * (d==1) # get probability of making observed choice (Bernoulli distribution)
    log_lik = np.log(choice_prob) # compute log-likelihood-contributions
    
    # 6. add log likelihood for mileage process
    if theta.size > 2: log_lik += compute_ll_p(model)[dx1]

    # 7. return
    if out == 1:
        return -np.mean(log_lik) # objective function (negative mean log likleihood)

    else:
        return model, lik_pr, pk, Vbar, dev, d, x, dx1

def score(theta, model, solver, data, pnames):

    global Vbar

    if theta.size > 2: 
        n_p = len(model.p)

    else:
        n_p = 0

    # 1. evaluate log-likelihood function
    model, lik_pr, pk, Vbar, dVbar, d, x, dx1 = ll(theta, model, solver, data, pnames, 9) 

    # 2. compute model derivatives in general form
    du_dtheta = compute_du_dtheta(model, n_p) # shape (n, number of parameters, 2)
    dVbar_dtheta = compute_dVbar_dtheta(model, pk, dVbar, n_p)

    dvkeep_dtheta = du_dtheta[..., 0] + model.beta * model.P1 @ dVbar_dtheta
    dvreplace_dtheta = du_dtheta[..., 1] + model.beta * model.P2 @ dVbar_dtheta
    dvalue_diff_x = dvkeep_dtheta - dvreplace_dtheta

    # 3. compute model derivatives for observations
    score_vec = (1-lik_pr-d)[:, None] * dvalue_diff_x[x, :]

    # 4. add derivative of log-likelihood from mileage process wrt. p
    if n_p > 0:

        # a. compute general model derivative
        dpll_dtheta = compute_dpll_dtheta(model, n_p)

        # b. compute model derivative for observations, i.e. slice relevant rows
        score_vec[...,2:] += dpll_dtheta[dx1]

    return score_vec

def grad(theta, model, solver,data, pnames):
    s = score(theta, model, solver, data,pnames)
    return -np.mean(s,axis=0)

def hes(theta, model, solver,data, pnames):
    " compute Hessian using Information Matrix Equality "
    s = score(theta, model, solver, data, pnames)
    return s.T@s/data.shape[0]

def updatepar(par,parnames, parvals):

    for i,parname in enumerate(parnames):
        # First two parameters are scalars
        if i<2:
            parval = parvals[i]
            setattr(par,parname,parval)

        else: # Remaining parameters are lists
            list_val = [None]*(parvals.size-2) 
            for j,parval in enumerate(parvals[2:]):
                list_val[j]=parval
            setattr(par,parname,list_val)

    return par

def compute_ll_p(model):

    # 1. add residual probability to p (we only compute frequincies for some "jumps")
    p = np.append(model.p,1-np.sum(model.p)) 

    # 2. penalize if model predicts negative probabilities
    if any(p<=0):
        log_lik_p = -100000*p 

    # 3. add log-likelihood contribution (multinomial distribution)
    else:
        log_lik_p = np.log(p)
    
    return log_lik_p

def compute_du_dtheta(model, n_p):

    dc = 0.001*model.grid # remember we have multiplied c with 0.001
    dutil_dtheta = np.zeros((model.n, 2 + n_p, 2)) # shape (n, number of parameters, number of choices)

    dutil_dtheta[:,0, 0] = 0 # derivative of keeping wrt RC
    dutil_dtheta[:,0, 1] = -1 # derivative of replacing wrt RC
    dutil_dtheta[:,1, 0] = -dc # derivative of keeping wrt c
    dutil_dtheta[:,1, 1] = -dc[0] # derivative of replacing wrt c

    return dutil_dtheta

def compute_dVbar_dtheta(model, pk, dVbar, n_p):
    
    # 1. compute Frechet derivative
    F = np.eye(model.n)-dVbar # shape (n, n)

    # 2. compute derivative of utility function wrt. parameters
    du_dtheta = compute_du_dtheta(model, n_p) # shape (n, number of parameters, 2)

    # 3. compute derivative of operator wrt. parameters in utility function
    p_keep = pk[:, None]
    p_replace = 1 - pk[:, None]
    dbellman_dtheta = p_keep * du_dtheta[..., 0] + p_replace * du_dtheta[..., 1] # shape (n, number of parameters)
    
    # 4. compute derivative of operator wrt. parameters that affect state transition
    if n_p > 0:
        dbellman_dtheta += model.beta*compute_dP_dtheta(model, Vbar, n_p, dbellman_dtheta.shape)

    # 5. compute derivative of Bellman wrt. parameters using implicit function theorem, see slides
    dVbar_dtheta = np.linalg.solve(F,dbellman_dtheta)

    return dVbar_dtheta

def compute_dP_dtheta(model, Vbar, n_p, shape):

    dP_dtheta = np.zeros(shape) # shape (n, number of parameters)

    for i_p in range(n_p): # loop over p
        
        # a. slice to get [Vbar(i_p), Vbar(i_p+1), ..., Vbar(n-1)], shape (n-i_p-1)
        direct = Vbar[i_p:-1] 

        # b. slice to get [Vbar(n_p), V_bar(n_p+1), ..., V_bar(n)], shape (n-n_p)
        Vbar_shifted = Vbar[n_p:]

        # c. repeat 
        Vbar_n = np.repeat(Vbar[-1], (n_p-i_p-1)) # shape (n_p-i_p-1)

        # d. stack columns
        indirect = np.hstack((Vbar_shifted, Vbar_n)) # n-n_p+n_p-i_p-1 = n-i_p-1

        dP_dtheta[:model.n-i_p-1, 2+i_p] = direct - indirect
    
    return dP_dtheta

def compute_dpll_dtheta(model, n_p):
    """ 
    Function that computes model based score of partial likelihood wrt. milage
    returns (n_p+1,n_p) matrix
    n_p + 1 rows since jump can be [0,1,...,n_p]
    n_p columns since there are only n_p parameters
    """

    # 1. compute numerical stable 1/p
    invp = np.exp(-np.log(model.p))

    # 2. compute diagonal 1/p
    diag_invp = np.diag(invp)

    # 3. compute -1/(p) for last row / residual probability
    ones_invp = -np.ones((1,n_p))*invp[n_p-1]

    # 4. stack in one array and return
    return np.vstack((diag_invp, ones_invp))