# load general packages
import numpy as np
import scipy.sparse as sparse
import scipy.optimize as optimize

# Import 
pk = 0.0 
def setup_data(data): 
   
    # setup
    class data_class: pass
    data_class.x = data.x 
    data_class.dx1= data.dx1
    data_class.dk = (data.d == 0)
    data_class.dr = (data.d == 1)
    return data_class

def estimate(model, data, Kmax = 100):
    global pk    

    # Find P
    tabulate = data.dx1.value_counts()
    p = [tabulate[i]/sum(tabulate) for i in range(tabulate.size-1)]
    model.p = p 
    model.state_transition()  
    pk0 = np.ones((model.n))*0.99  # starting value for CCP's

    theta0 = [0,0]     

    for K in range(Kmax):

        # Step 0)  Pre-compute unconditional transition matrix Fu and the inverse of I-beta*Fu to used in step 1 
        model.unc_state_transition(pk0) # Calculate Fu and Finv

        # Step 1)  Maximize the pseudo-likelihood function given step K-1 CCPs
        res = optimize.minimize(ll,theta0,args =(model, data, pk0), method='Newton-CG', jac = grad, hess= hes, tol = 1e-6)
        theta_hat = res.x
        NPL_metric = np.abs(theta0-theta_hat) 


        # Step 2)  Update CCPs using theta_npl from step 1)
        pk0 = pk
        theta0 = theta_hat
        if NPL_metric.all() < 1e-6:
            return res, theta_hat,pk
    
    print(f'The function did not converge after {K} iterations')
    
    return res, theta_hat, pk

def ll(theta, model, data, pk0,out=1):
    global pk

    # update parameters
    model.RC = theta[0]
    model.c = theta[1]
    model.create_grid()

    # Solve the model
    pk = model.psi(pk0,model.Finv)

    pKdata = pk[data.x] 

    if out == 2:
        return pk, pKdata

    log_lik = np.log(data.dk*pKdata+(1-pKdata)*data.dr) 
    f = -np.mean(log_lik)
    return f

def score(theta, model, data, pk0):
    global pk
    pk, pkdata = ll(theta, model, data, pk0,out=2)
    res = data.dk-pkdata

    NT = ((pkdata.size))
    score = np.zeros((NT,theta.size))
    dP = model.P1[0,:]-model.P1  

    dvdRC = np.ravel(-1+model.beta*dP@model.Finv@(1-pk[:,None])*(-1))
    dvdc = np.ravel(model.dc + model.beta*dP@model.Finv@(pk*(-model.dc)))
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
    pk0 = np.ones((model.n))*0.99  # starting value for CCP's
    pk = np.nan+np.zeros((100,model.n))
    pk[0,:] = pk0
    model.create_grid()


    for i in range(1,100): 
        pk[i,:]  = model.psi(pk[i-1,:])    
    return pk
