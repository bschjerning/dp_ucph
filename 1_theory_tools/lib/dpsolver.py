import numpy as np
from scipy.stats import lognorm
from time import process_time
class dpsolver():
    '''A class for solving dynamic programming models'''
    def vfi_T(model, T=100, callback=None):
        '''Solves the model using backward induction (VFI with maxiter =T)
        Parameters:
            T: number of periods to solve
            callback: function to call after each iteration
        Returns:
            V: value function
            policy: policy functions (p1, p2, ..., pN)'''
        tic = process_time() # Start the stopwatch / counter
        V=np.zeros((model.n_x, T+1)) # on first iteration assume consuming everything
        policy = np.zeros((model.n_x, model.n_choices, T+1))  # Stores policy functions (p1, p2, ..., pn_choices)'''

        for t in range(T-1, 0, -1):
            V[:,t-1],policy[:,:,t-1]=model.bellman(V[:,t])
            if callback: callback(t,model.x,V, policy) # callback for making plots and plotting iterations
        else:  # when i went up to maxiter
            toc = process_time() # Stop the stopwatch / counter
            print('Solved by backward induction using',round(toc-tic, 5), 'seconds')
        return V,policy
    
    def vfi(model, maxiter=100, tol=1e-6,callback=None):
        '''Solves the model using VFI (successive approximations)
        Parameters:
            maxiter: maximum number of iterations
            tol: tolerance for convergence
            callback: function to call after each iteration
            Returns:
            V: value function
            policy: policy functions (p1, p2, ..., p_n_choices)'''
        tic = process_time() # Start the stopwatch / counter
        V0=np.zeros(model.n_x) # on first iteration assume consuming everything
        for iter in range(maxiter):
            V1,policy=model.bellman(V0)
            if callback: callback(iter,model.x,V1, policy, V0) # callback for making plots
            if np.max(abs(V1-V0)) < tol:
                toc = process_time() # Stop the stopwatch / counter
                print('Solved by VFI in', iter, 'iterations using',round(toc-tic, 5), 'seconds')
                break
            V0=V1
        else:  # when i went up to maxiter
            print('No convergence: maximum number of iterations achieved!')
        return V1,policy
    
    def iterinfo(iter,model,V1,c=None, V0=0):
        print('iter=', iter, '||V1-V0||', np.max(abs(V1-V0)))
    