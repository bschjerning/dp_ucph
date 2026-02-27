# load general packages
import numpy as np
import scipy.sparse as sparse
import scipy.optimize as optimize

pk = 0.0 

def setup_data(data): 
   
    # setup data class
    class data_class: pass
    data_class.x = data.x -1
    data_class.dx1 = data.dx1
    data_class.d = data.d
    return data_class

def estimate(model, data, Kmax = 100):
    '''Estimate model using Nested Psuedo Likelihood (NPL)'''

    # 1. load previous choice probability as global variable
    global pk    

    # 2. find transition probabilities p (non-parametrical 1-step estimator)
    tabulate = data.dx1.value_counts() # Count number of observations for each dx1
    p = [tabulate[i]/sum(tabulate) if i < len(tabulate) else 0 for i in range(len(model.p))]
    model.p[:] = p
    
    # 3. compute state transition matrix given p guess
    model.state_transition()  
    
    # 4. set starting valiues for policy iteration, guess on policies are all 0.99
    pk0 = np.ones((model.n))*0.99

    # 5. guess on parameters
    theta0 = [0,0]

    # 6. outer loop: update CCPs until convergence or Kmax iterations are reached
    for _ in range(Kmax):

        # a. pre-compute unconditional transition matrix Fu and the inverse of I-beta*Fu to be used in b)
        model.unc_state_transition(pk0)

        #Inner loop:
        # Step 1)  Maximize the pseudo-likelihood function given step K-1 CCPs
        res = optimize.minimize(ll, theta0, args = (model, data, pk0), method='Newton-CG', jac = grad, hess = hes, tol = 1e-6)
        theta_hat = res.x
        NPL_metric = np.abs(theta0-theta_hat) #save distance between parameters

        #Outer loop step
        # Step 2)  Update CCPs using theta_npl from step 1)
        pk0 = pk
        theta0 = theta_hat
        if NPL_metric.all() < 1e-6: # check convergence
            return res, theta_hat,pk
    
    print(f'The function did not converge after {Kmax} iterations')
    
    return res, theta_hat, pk

def ll(theta, model, data, pk0, out = 1):
    '''Log-likelihood function for NPL'''

    # 1. load previous choice probability as global variable
    global pk

    # 2. update parameters
    model.RC = theta[0]
    model.c = theta[1]
    model.create_grid()

    # 3. update CCPs
    # FILL IN.
    pk = None

    ### SOLUTION ###
    pk = model.Psi(pk0, Finv = model.Finv)
    ### SOLUTION ###
    
    # 4. map choice probabilities to data
    pk_data = pk[data.x]

    # return CCPs if out = 2
    if out == 2:
        return pk, pk_data

    # 5. Calculate log-likelihood
    # FILL IN.
    log_lik = None

    ### SOLUTION ###
    log_lik = (data.d == 0) * np.log(pk_data) + (data.d == 1) * np.log(1-pk_data)
    ### SOLUTION ###
    
    # 6. return log-likelihood if out = 1
    f = -np.mean(log_lik)
    return f

def score(theta, model, data, pk0):

    # 1. load previous choice probability as global variable
    global pk

    # 2. call log-likelihood function
    pk, pk_data = ll(theta, model, data, pk0,out=2)

    # 3. compute "error" / residual term
    res = 1*(data.d == 0)-pk_data

    # 4. allocate
    score = np.zeros((pk_data.size, theta.size))

    dP = model.P2-model.P1  
    pr = (1-pk[:, None]) # shape (n,1)

    # 5. compute utility derivatives
    dureplace_dRC = -1
    dukeep_dc = -model.dc

    # 6. compute operator derivatives
    dbellman_dRC = pr * dureplace_dRC
    dbellman_dc = pk * dukeep_dc

    # 7. compute Vbar derivatives using implicit function theorem
    dVbar_dRC = model.Finv @ dbellman_dRC
    dVbar_dc = model.Finv @ dbellman_dc

    # 8. compute difference-in-value-choice derivative wrt. RC and c
    dvdRC = (dureplace_dRC + model.beta * dP @ dVbar_dRC).flatten()
    dvdc = (-dukeep_dc + model.beta * dP @ dVbar_dc).flatten()

    # 9. compute score and store
    score[:,0] = res*dvdRC[data.x]
    score[:,1] = res*dvdc[data.x]

    return score 

def grad(theta, model, data, pk0):
    s = score(theta, model, data, pk0)
    return np.mean(s,0)

def hes(theta, model, data, pk0):
    s = score(theta, model, data, pk0)
    return s.T@s/data.x.shape[0]

def solve(model):
    '''Solve model by successive approximation in choice probabilitiy space (policy iteration)'''

    # 1. set starting valiues
    pk0 = np.ones((model.n))*0.99 
    
    # 2. allocate for 100 iterations
    iterations = 100
    pk = np.zeros((iterations, model.n)) + np.nan
    pk[0,:] = pk0
    
    # 3. update grids
    model.create_grid()

    # 4. solve using successive approximations
    for i in range(1, iterations): 
        pk[i,:]  = model.Psi(pk[i-1,:])   

    return pk
