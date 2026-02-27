# Zurcher class: Contains model parts for Rust's engine repplacement model Rust(Ecta, 1987)

#import packages
import numpy as np
import time
import pandas as pd

class zurcher():

    def __init__(self,**kwargs):

        self.setup(**kwargs)

    def setup(self,**kwargs):     
  
        # 1. parameters
        self.n = 175 # number of grid points
        self.max = 450 # max mileage

        # 2. structual parameters (to be estimated in NFXP)
        self.p = np.array([0.0937, 0.4475, 0.4459, 0.0127])   # transition probability
        self.RC = 11.7257                                     # replacement cost
        self.c = 2.45569                                      # cost parameter
        self.beta = 0.9999                                    # discount factor

        # 3. update baseline parameters using keywords
        for key,val in kwargs.items():
            setattr(self,key,val) 

        # 4. Create grid
        self.create_grid()

    def create_grid(self):

        # 1. construct milage grid
        self.grid = np.arange(0,self.n)

        # 2. construct cost grid associated with each point in mileage grid
        self.cost = 0.001*self.c*self.grid

        # 3. construct Markov (transition) matrices
        self.state_transition() 

    def state_transition(self):
        '''
        Compute transition probability matrixes conditional on choice
        '''

        # 1. get transition probabilities, last one is residual
        p = np.append(self.p,1-np.sum(self.p))

        # 2. initialize Markov (transition) matrix for keeping (d = 0)
        P1 = np.zeros((self.n,self.n))

        # 3. fill out
        for i in range(self.n): # loop over rows
            if i <= self.n-len(p): # check if p vector fits entirely
                P1[i,i:i+len(p)]=p
            else:
                P1[i,i:] = p[:self.n-len(p)-i]
                P1[i,-1] = 1.0-P1[i,:-1].sum()

        # 4. initialize Markov (transition) matrix for replacing (d = 1)
        P2 = np.zeros((self.n,self.n))
        for i in range(self.n): # loop over rows
            P2[i][:len(p)]=p

        # 5. store
        self.P1 = P1
        self.P2 = P2

    def bellman(self,Vbar,output=1):
        '''
        Evaluate Bellman operator, choice probability and Frechet derivative
        (written in integrated value form, like simple zurcher, last week)
        output = 1: return only integrated value function
        output = 2: + choice probabilities
        output = 3: + Frechet derivative
        '''

        # 1. compute value-choice functions using cost-grids and Markov matrices
        value_keep = -self.cost + self.beta * self.P1 @ Vbar # shape (n,)
        value_replace = -self.RC - self.cost[0] + self.beta * self.P2 @ Vbar # shape (n,)

        # 2. recenter Bellman by subtracting max(value_keep, value_replace) for numerical stability
        maxV = np.maximum(value_keep, value_replace) # element-wise maximum

        # 3. compute logsum to handle expectation over unobserved states / taste shocks
        logsum = (maxV + np.log(np.exp(value_keep-maxV)  +  np.exp(value_replace-maxV))) 
        Vbar = logsum # Bellman operator as integrated value

        # 4. compute choice probabilities and Frechet derivative -> return
        if output == 1:
            return Vbar
        
        pk = 1/(1+np.exp(value_replace-value_keep))  
        if output == 2:
            return Vbar, pk

        dev1 = self.dbellman(pk)
        return Vbar, pk, dev1
    
    def dbellman(self,pk): 
        '''
        Compute derivative of Bellman operator wrt. Vbar
        '''

        # 1. unpack Markov (transition) matrices, shapes (n,n)
        P_keep = self.P1
        P_replace = self.P2

        # 2. diagonal matrices, shapes (n,n)
        pk_diag = np.diag(pk)
        pr_diag = np.diag(1-pk)

        # 3. compute derivative
        dev1 = self.beta * (pk_diag @ P_keep + pr_diag @ P_replace)

        return dev1
    
    def read_busdata(self, bustypes=[1,2,3,4]):

        # 1. load raw data from CSV
        df = pd.read_csv("busdata1234.csv", header=None)

        # 2. rename relevant columns
        df = df.rename(columns={
            0: "id",          # bus ID
            1: "bustype",     # bus type
            4: "d_lag",       # lagged replacement dummy
            6: "x_raw"        # raw odometer reading
        })

        # 3. construct current replacement dummy (lead of d_lag within each bus)
        df["d"] = df.groupby("id")["d_lag"].shift(-1).fillna(0)

        # 4. discretize odometer into grid points 1,...,n
        df["x"] = (df["x_raw"] * self.n / (self.max * 1000)).apply(np.ceil)

        # 5. compute lagged mileage within each bus
        df["x_lag"] = df.groupby("id")["x"].shift(1).fillna(0)

        # 6. compute monthly mileage change
        df["dx1"] = df["x"] - df["x_lag"]

        # 7. if engine was replaced last period, mileage jump equals current state
        df.loc[df["d_lag"] == 1, "dx1"] = df["x"]

        # 8. cap mileage jumps at the grid size
        df["dx1"] = df["dx1"].clip(upper=len(self.p))

        # 9. convert state variables to integers
        df["x"] = df["x"].astype(int)
        df["dx1"] = df["dx1"].astype(int)

        # 10. remove first observation per bus (missing lagged mileage)
        df = df[df.groupby("id").cumcount() != 0]

        # 11. keep only selected bus types
        df = df[df["bustype"].isin(bustypes)]

        # 12. return final dataset
        return df[["d", "x", "dx1"]]

    def sim_data(self,N,T,pk): 

        # 1. set seed
        np.random.seed(2026)
        
        # 2. indices
        idx = np.tile(np.arange(1,N+1),(T,1))  
        ts = np.tile(np.arange(1,T+1),(N,1)).T
        
        # 3. draw random numbers
        u_init = np.random.randint(self.n,size=(1,N)) # initial condition
        u_dx = np.random.rand(T,N) # uniform distribution on [0,1), shape (T, N)
        u_d = np.random.rand(T,N) # uniform distribution on [0,1), shape (T, N)

        # 4. find states and choices
        csum_p = np.cumsum(self.p)
        dx1 = 0

        # 5. find milage shock by using uniform shock
        for val in csum_p:
            dx1 += u_dx > val
        
        # 6. allocate 
        x = np.zeros((T,N), dtype = int)
        x1 = np.zeros((T,N), dtype = int)
        d = np.zeros((T,N)) + np.nan

        # 7. set initial state
        x[0,:] = u_init

        # 8. loop over periods forwards
        for t in range(T):

            # a. if u_d (uniform in [0,1) ) is below probability of replacing, then replace
            d[t,:] = u_d[t,:] < 1 - pk[x[t,:]] 

            # b. compute state transition with minimum to avoid exceeding the maximum mileage
            x1[t,:] = np.minimum(x[t,:]*(1-d[t,:]) + dx1[t,:] , self.n-1)

            # c. if we are not in last period of simulation -> state transition in x
            if t < T-1:
                x[t+1,:] = x1[t,:]
                
        # 9. reshape, the F-order is to ensure long-format when making the pandas df
        idx = np.reshape(idx,T*N, order='F')
        ts = np.reshape(ts,T*N, order='F')
        d = np.reshape(d,T*N, order='F')
        x = np.reshape(x,T*N, order='F') + 1 # add 1 to make index start at 1 as in data - 1,2,...,n
        x1 = np.reshape(x1,T*N, order='F') + 1 # add 1 to make index start at 1 as in data - 1,2,...,n
        dx1 = np.reshape(dx1,T*N, order='F')

        # 10. construct df and return
        data = {'id': idx,'t': ts, 'd': d, 'x': x, 'x1': x1, 'dx1': dx1}
        df= pd.DataFrame(data) 

        return df

    def eqb(self, pk):
        # Inputs
        # pk: choice probability

        # Outputs    
        # pp: Pr{x} (Equilibrium distribution of mileage)
        # pp_K: Pr{x,i=Keep}
        # pp_R: Pr{x,i=Replace}
        pl = self.P1 * pk[:,None] + self.P2 * (1 - pk[:,None])

        pp = self.ergodic(pl)

        # joint probabilities
        # a. joint probability of x and i=keep
        pp_K = pk * pp
        # b. joint probability of x and i=replace
        pp_R = (1 - pk) * pp

        return pp, pp_K, pp_R

    def ergodic(self,p):
        #ergodic.m: finds the invariant distribution for an NxN Markov transition probability: q = qH , you can also use Succesive approximation
        n = p.shape[0]
        if n != p.shape[1]:
            print('Error: p must be a square matrix')
            ed = np.nan
        else:
            ap = np.identity(n)-p.T
            ap = np.concatenate((ap, np.ones((1,n))))
            ap = np.concatenate((ap, np.ones((n+1,1))),axis=1)

            # find the number of linearly independent columns
            temp, _ = np.linalg.eig(ap)
            temp = ap[temp==0,:]
            rank = temp.shape[1]
            if rank < n+1:
                print('Error: transition matrix p is not ergodic')
                ed = np.nan
            else:
                ed = np.ones((n+1,1))
                ed[n] *=2
                ed = np.linalg.inv(ap)@ed
                ed = ed[:-1]
                ed = np.ravel(ed)

        return ed