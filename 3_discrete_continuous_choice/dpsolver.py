from time import process_time
import numpy as np
class dpsolver():
    '''Class to implement deaton's model with log-normally distrubuted income shocks'''
    def vfi_T(model, T=100, callback=None):
        '''Solves the model using backward induction (VFI with maxiter =T'''
        tic = process_time() # Start the stopwatch / counter
        V=np.zeros((model.n_x, T+1)) # on first iteration assume consuming everything
        c=np.zeros((model.n_x, T+1)) # on first iteration assume consuming everything
        for t in range(T-1, 0, -1):
            V[:,t-1],c[:,t-1]=model.bellman(V[:,t])
            if callback: callback(t,model.x,V, c) # callback for making plots and plotting iterations
        else:  # when i went up to maxiter
            toc = process_time() # Stop the stopwatch / counter
            print(model, 'solved by backward induction using',round(toc-tic, 5), 'seconds')
        return V,c
    
    def vfi(model, maxiter=100, tol=1e-6,callback=None):
        '''Solves the model using VFI (successive approximations)'''
        tic = process_time() # Start the stopwatch / counter
        V0=np.zeros(model.n_x) # on first iteration assume consuming everything
        for iter in range(maxiter):
            V1,c1=model.bellman(V0)
            if callback: callback(iter,model.x,V1, c1, V0) # callback for making plots
            if np.max(abs(V1-V0)) < tol:
                toc = process_time() # Stop the stopwatch / counter
                print(model, 'solved by VFI in', iter, 'iterations using',round(toc-tic, 5), 'seconds')
                break
            V0=V1
        else:  # when i went up to maxiter
            print('No convergence: maximum number of iterations achieved!')
        return V1,c1
    
    def iterinfo(iter,model,V1,c=None, V0=0):
        print('iter=', iter, '||V1-V0||', np.max(abs(V1-V0)))