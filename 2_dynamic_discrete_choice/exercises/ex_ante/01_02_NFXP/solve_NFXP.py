# NFXP class: Solves Rust's engine repplacement model Rust(Ecta, 1987) 

# import packages
import numpy as np
import time

class solve_NFXP():

    def __init__(self,**kwargs):

        self.setup(**kwargs)

    def setup(self,**kwargs):
        """
        Function sets up the default values for the NFXP algorithm.
        If kwargs exist, update the default values with the input values.
        """
 
        # 1. sa / vfi steps
        self.sa_max = 50  # maximum number of contractions steps
        self.sa_min = 10  # minimum number of contraction steps
        self.sa_tol = 1.0e-10  # absolute tolerance before

        # 2. newton-kantorovich steps
        self.max_fxpiter = 5  # maximum number of times to switch between newton-kantorovich iterations and contraction iterations.
        self.pi_max = 10  # maximum number of newton-kantorovich steps

        self.pi_tol = 1.0e-12  # final exit tolerance in fixed point algorithm, measured in units of numerical precision
        self.tol_ratio = 1.0e-02  # relative tolerance before switching to n-k algorithm when discount factor is supplied as input in solve.poly
        self.printfxp = 0  # print iteration (0=no output), (1=compressed output), (2=detailed iteration output)

        # 3. if kwargs exist: update the default values with the input values
        for key, val in kwargs.items():
            setattr(self, key, val)

    def poly(self,bellman, V0=np.zeros(1), beta= 0.0, output=1):
        """
        Solves the model using the poly-algorithm.
        set beta = 0.0 if you want to solve only with successive approximations.
        """

        # 1. set starting time
        t0poly = time.time()

        # 2. loop over the maximum number of switches between SA and NK iterations
        for k in range(self.max_fxpiter):

            # a. start with SA
            if self.printfxp > 0: print(f'\nBeginning SA-iterations (for the {k+1} time)\n')
            V0,iter_sa= self.sa(bellman,V0,beta)

            # b. then NK-iterations
            if self.printfxp>0: print(f'\nBeginning NK-iterations (for the {k+1} time)\n')
            V0,pk,dV, iter_nk = self.nk(bellman,V0)

            # c. time it
            t1poly = time.time()

            if iter_nk.converged=='true':
                if self.printfxp>0:
                    print(f'\nConvergence achieved!')
                    print(f'\nElapsed time: {(t1poly-t0poly):.4f} (seconds)')
                    break 
        
            else:
                if k >= self.max_fxpiter:
                    print(f'No convergence! Maximum number of iterations exceeded without convergence!')
                    break

        # 3. converged -> return final V0
        V = V0

        # 4. user-specified output
        if output==1:            
            return V
        if output==2:            
            return V, pk
        if output==3:            
            return V, pk, dV
        if output==5:            
            return V, pk, dV, iter_sa, iter_nk
        else:
            print('solve_NFXP.poly: output must be 1,2,3 or 5')

    def sa(self,bellman,V0=np.zeros(1), beta=0.0):
        """
        Function performs a number of contraction steps
        """

        # 1. set starting time
        t0 = time.time()

        # 2. empty class to store the iteration output
        class iteration: pass
        iteration.tol = np.zeros((self.sa_max)) + np.nan
        iteration.rtol = np.zeros((self.sa_max)) + np.nan
        iteration.converged = 'false'

        # 3. loop over the maximum number of contraction iterations
        for i in range(self.sa_max):

            # a. do contraction step
            V = bellman(V0,output=1) 

            # b. compute max change in value functions from contraction step, standard "measure" of progress, as earlier in course
            iteration.tol[i] = max(abs(V-V0))

            # c. update guess on value function
            V0 = V.copy()

            # d. evaluate stopping criteria 1: check if tolerance is below the specified tolerance by:
            # i) find max value ii) compute log10 of it iii) return integer of it when rounding up
            adj  = np.ceil(np.log10(abs(max(V0)))) # "how many zeros (or decimals) on number"

            # compute tolerance in levels, accomodates that numerically small value functions should e.g. have lower tolerance
            ltol = self.sa_tol*10**adj # 10**adj/(10^{-10})

            # check tolerance
            if (i>=self.sa_min) and (iteration.tol[i]<ltol):
                iteration.message = f'\nSA converged after {i} iterations, tolerance: {iteration.tol[i]:.4f}'
                iteration.converged = 'true'
                break

            # e. evaluate stopping criteria 2: check relative tolerance and switch to NK algorithm
            if i>=self.sa_min:
                iteration.rtol[i] = iteration.tol[i]/iteration.tol[max(i-1,0)] # 
                if (abs(beta-iteration.rtol[i]) < self.tol_ratio): # if relative tolerance is "close" to beta -> stop and switch to NK
                    iteration.message = '\nSA stopped prematurely due to relative tolerance. Start NK iterations'
                    iteration.converged = 'halfway'
                    break

        # 4. stop timer
        t1 = time.time()
        
        # 5. store iteration output, print, return
        iteration.n = i+1
        iteration.tol = iteration.tol[0:i+1]
        iteration.rtol = iteration.rtol[0:i+1]
        iteration.time = t1-t0 

        self.print_output(iteration)

        return V, iteration

    def nk(self,bellman, V0):
        """
        Solves the model using the Newton-Kantorovich steps
        """

        # 1. set starting time
        t0 = time.time()

        # 2. empty class to store the iteration output
        class iteration: pass
        iteration.tol = np.zeros((self.pi_max)) + np.nan
        iteration.rtol = np.zeros((self.pi_max)) + np.nan
        iteration.converged = 'false'

        # Get the state space size
        m = V0.size

        # 3. loop over the maximum number of Newton-Kantorovich steps
        for i in range(self.pi_max):

            # a. compute derivative of bellman
            V1, pk, dV = bellman(V0,output=3)

            # b. compute Frechet derivative
            F = np.eye(m)-dV

            # c. do NK-step
            V = V0 - np.linalg.inv(F) @ (V0 - V1)
            
            # d. do SA iteration for stability and accurate measure of error bound
            V0 = bellman(V,output=1)

            # e. evaluate stopping criteria
            # compute tolerance
            iteration.tol[i]=max(abs(V-V0))
            iteration.rtol[i] = iteration.tol[i]/(iteration.tol[max(i-1,0)] + 1.0e-15) # avoid zero-divison   

            # f. compute tolerance in level (see SA function for explanation)
            adj  = np.ceil(np.log10(abs(max(V0))))
            ltol = self.pi_tol*10**adj

            if iteration.tol[i] < ltol:
                iteration.message = f'\nNK converged after {i+1} iterations, tolerance: {iteration.tol[i]:.4f}'
                iteration.converged = 'true'
                break

        # 4. stop timer
        t1 = time.time()
    
        # 5. store iteration output, print and return
        iteration.n = i+1
        iteration.tol = iteration.tol[0:i+1]
        iteration.rtol = iteration.rtol[0:i+1]
        iteration.time = t1-t0 

        self.print_output(iteration)

        return V, pk, dV, iteration

    def print_output(self,iteration):

        if self.printfxp>1: # print detailed output
            for i,iter_tol in enumerate(iteration.tol):
                print(f'Iteration {i+1}\ttol: {iter_tol:.4f}\ttol(j)/tol(j-1): {iteration.rtol[i]:.4f}')

        if self.printfxp>0: # print final output
            if iteration.converged != 'false':
                print(f'{iteration.message}')
            else:
                print(f'\nMaximum number of iterations reached, tolerance: {iteration.tol[-1]:.4f}')
            print(f'\nElapsed time {iteration.time:.4f} seconds')
        