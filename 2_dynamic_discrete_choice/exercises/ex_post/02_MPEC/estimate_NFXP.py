#  NFXP class for structural estimation of discrete choice models.

import numpy as np
import scipy.optimize as optimize


ev = np.zeros(1) # Global variable

def estimate(model,solver,data,theta0=[0,0],twostep=0):
    global ev
    ev = np.zeros(1) 
    
    samplesize = data.shape[0]
    # STEP 1: Find p 
    tabulate = data.dx1.value_counts()
    p = [tabulate[i]/sum(tabulate) for i in range(tabulate.size-1)]

    # STEP 2: Estimate structual parameters
    model.p = p # Use first step estimates as starting values for p
    
    # Estimate RC and C
    pnames = ['RC','c']
    
    res = optimize.minimize(ll,theta0,args = (model, solver, data, pnames), method = 'trust-ncg',jac = grad, hess = hes, tol=1e-8)
    model=updatepar(model,pnames,res.x)
    
    # Estimate RC, c and p
    if twostep == 0:
        pnames = ['RC','c','p']
        theta0 = [model.RC, model.c] + model.p.tolist()
        res = optimize.minimize(ll,theta0, args = (model,solver,data, pnames), method = 'trust-ncg',jac = grad, hess = hes, tol = 1e-8)

        model=updatepar(model,pnames,res.x)

    # Converged
    converged   =   (res.status == 2 or res.status ==0)

    # Compute Variance-Covaiance matrix
    h = hes(res.x, model, solver,data, pnames)
    Avar = np.linalg.inv(h*samplesize)

    theta_hat = res.x
    
    return model, res, pnames, theta_hat, Avar, converged

def ll(theta, model, solver,data, pnames, out=1): # out=1 solve optimization
    global ev
    
    # Unpack
    x = data.x
    d = data.d
    dx1 = data.dx1

    # Update values
    model=updatepar(model,pnames,theta)
    model.p = np.abs(model.p)    # helps BHHH which is run as unconstrained optimization
    model.create_grid()
    ev0 = ev

    # Solve the model
    ev, pk, dev = solver.poly(model.bellman, V0=ev0 ,beta=model.beta, output=3)

    # Evaluate likelihood function
    lik_pr = pk[x]    
    log_lik = np.log(lik_pr+(1-2*lik_pr)*d)       
    
    # add on log like for mileage process
    if theta.size>2:
        p = np.append(model.p,1-np.sum(model.p))
        if any(p<=0):
            log_lik -= 100000*p[dx1]
        else:
            log_lik += np.log(p[dx1])
        
    else:
        p = np.nan


    if out == 1:
        # Objective function (negative mean log likleihood)
        return np.mean(-log_lik)

    return model,lik_pr, pk, ev, dev, d,x,dx1

def score(theta, model, solver, data, pnames):
    global ev
    model,lik_pr, pk, ev, dev, d,x,dx1 = ll(theta, model, solver, data, pnames,9)
    F = np.eye(model.n)-dev    
    N = data.x.size
    dc = 0.001*model.grid

    # Compute the score
    if theta.size>2:
        n_p = len(model.p)
    else:
        n_p = 0

    # Step 1: compute derivative of contraction operator wrt. parameters
    dbellman_dtheta=np.zeros((model.n,2 + n_p)) 
    dbellman_dtheta[:,0] = (1-pk)*(-1) # derivative wrt RC
    dbellman_dtheta[:,1] = pk*(-dc)   # derivative wrt c

    if theta.size>2:        
        vk = -model.cost+model.beta*ev
        vr = -model.RC-model.cost[0]+model.beta*ev[0]
        vmax = np.maximum(vk,vr)
        dbellman_dpi = vmax+np.log(np.exp(vk-vmax)+np.exp(vr-vmax))

        for i_p in range(n_p):
            part1 = dbellman_dpi[i_p:-1]
            part2 = np.hstack((dbellman_dpi[n_p:model.n], np.tile(dbellman_dpi[-1],(n_p-i_p-1))))
            dbellman_dtheta[0:model.n-i_p-1,2+i_p] =part1-part2
        
        invp=np.exp(-np.log(model.p))
        invp = np.vstack((np.diag(invp[0:n_p]),-np.ones((1,n_p))*invp[n_p-1]))
      
    # Step 2: compute derivative of ev wrt. parameters
    dev_dtheta = np.linalg.solve(F,dbellman_dtheta)

    # Step 3: compute derivative of log-likelihood wrt. parameters
    score = ((d - (1- lik_pr))[:,None])   * ( np.vstack((-np.ones(N), dc[x-1], np.zeros((n_p,N)))).T + np.broadcast_to(dev_dtheta[0],(N,2+n_p)) - dev_dtheta[x-1] )

    if theta.size>2:
        for i_p in range(n_p): 
            score[:,2+i_p] = score[:,2+i_p]+invp[dx1,i_p]

    return score

def grad(theta, model, solver,data, pnames):
    s = score(theta, model, solver, data,pnames)
    return -np.mean(s,0)

def hes(theta, model, solver,data, pnames):
    s= score(theta, model, solver, data, pnames)

    return s.T@s/data.shape[0]

def updatepar(par,parnames, parvals):

    for i,parname in enumerate(parnames):
        if i<2:
            parval = parvals[i]
            setattr(par,parname,parval)
        else:
            list_val = [None]*(parvals.size-2) 
            for j,parval in enumerate(parvals[2:]):
                list_val[j]=parval
            setattr(par,parname,list_val)
    return par

