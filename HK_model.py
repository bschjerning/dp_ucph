###### Load Packages ########
from ast import Raise
from ssl import VERIFY_CRL_CHECK_CHAIN
from types import SimpleNamespace
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from scipy import optimize, stats
import pandas as pd
import os
import pdb
from scipy.sparse import csr_matrix



##### Define Model Object ######
class model_HK():
    def __init__(self,**kwargs):
        self.setup(**kwargs)
    
    def setup(self, **kwargs):
        #Set defaults
        self.gamma = 0 # CRRA parameter
        self.beta = 0.95 #Discount factor
        self.delta_0 = 0 #Constant in unemployment
        self.delta_1 = 6 # Constant in utility
        self.delta_2 = 21.599999999999998 # Constant in wage
        self.kappa_d0 = 0  #Lagged dummy in unemployment
        self.kappa_G1 = 0  #Ability parameter
        self.kappa_G2 = 0  # Ability parameter
        self.kappa_H1  = 0 # cutoff in unemployment
        self.phi_H1 = 6 # Cutoff 1 in education
        self.phi_H2 = 0 # Cutoff 2 in education
        self.phi_H3 = 0 # Cutoff 3 in education
        self.phi_G1 = 0 # Ability parameter
        self.phi_G2 = 0 # Ability parameter
        self.phi_d0 = 0 # dummy in education
        self.r = 1 #Labor wage scale parameter
        self.alpha_H1 = 1.7999999999999998 # Human capital wage reward
        self.alpha_H2 = 0 # Human capital wage reward
        self.alpha_K1 = 1.2 #Work experience wage reward
        self.alpha_K2 = -0.06 #Work experience wage reward
        self.alpha_A1 = 0.0 #Age wage reward
        self.alpha_A2 = -0.0 #Age wage reward
        self.alpha_G1 = 3.0 #Ability/Grade wage reward
        self.alpha_G2 = 2.0 #Ability/Grade reward
        self.A_dummy = 0 # Age parmater
        self.nd = 3 #Number of possible choices
        self.nT = 15 #Number of periods
        self.nH = 27 # Number of human capital grid points
        self.nK = 31 #Number of work experience grid points
        self.A_min = 18 #Minimum age
        self.A_max = 18 #Maximum age
        self.lagd_c = 2 #Not used
        self.nG = 3 #Number of grade grid points
        self.N = 100 #Number of agents in simulation
        self.eps_sigma = 1 #Std of choice-specific shocks
        self.nu_sigma = 1 #Wage error std
        self.pnames = ['delta_0','delta_1','delta_2','phi_H1','alpha_H1','alpha_H2','alpha_K1','alpha_K2']
        # b. update baseline parameters using keywords
        for key,val in kwargs.items():
            setattr(self,key,val) 
        self.nA = self.A_max - self.A_min + 1 #Number of age grid points
        #Create grids
        #self.create_grids()

    def create_grids(self):
        ######## Initialize and create grids for later use #########
        self.dgrid = np.arange(0,self.nd) #Choice grid
        self.Hgrid = np.arange(0, self.nH) # Human capital grid
        self.Kgrid = np.arange(0, self.nK) # Human capital grid
        self.Agrid = np.arange(self.A_min, self.A_max + 1) # Human capital grid
        self.Ggrid = np.arange(0, self.nG) # Human capital grid
        self.sim = SimpleNamespace()
        self.sim.H = np.zeros((self.nT, self.N), dtype=int)
        self.sim.K = np.zeros((self.nT, self.N), dtype=int)
        self.sim.A = np.zeros((self.nT, self.N), dtype=int)
        self.sim.dlag = np.zeros((self.nT, self.N), dtype=int)
        self.sim.G = np.zeros((self.nT, self.N), dtype=int)
        self.sim.X = np.zeros((self.nT, self.N), dtype=int)
        self.sim.Z = np.zeros((self.nT, self.N), dtype=int)
        self.sim.d = np.zeros((self.nT, self.N), dtype=int)  
        self.sim.eps = np.zeros((self.nT, self.N), dtype=float)
        self.sim.w = np.zeros((self.nT, self.N), dtype=float)
        self.state_transition() # Compute state-transition matrix
        

    def _tensorproduct(self,*args):
        ####### Creates tensor product of transition matrices ########
        ''' 
        Input: Transitions matrices of state variables
        '''
        #Initiate new matrix
        Xtrans = args[0].copy()
        nX = Xtrans.shape[0] #number of grid points
        nd = Xtrans.shape[2] #Number of choices

        for Atrans in args[1:]:
            nA = Atrans.shape[0]
            Xtrans_temp = np.zeros([nX*nA,nX*nA,nd])
            for d in range(nd): #Number of choices
                Xtrans_temp[:,:,d] = np.kron(Atrans[:,:,d],Xtrans[:,:,d])
            Xtrans = Xtrans_temp
            nX *= nA

        #Make Xtrans sparse
        Xtrans_dict = {}
        for d in range(nd):
            #Xtrans_dict[d] = Xtrans[:,:,d] #Not sparse
            Xtrans_dict[d] = csr_matrix(Xtrans[:,:,d]) #Sparse
        #Note that Xtrans_dict other places will be referred to as Xtrans

        return nX, Xtrans_dict, #Xtrans_cumsum

    def state_transition(self):
        ######### Contruct transition matrices ######
        #State transition in H
        self.Htrans = np.zeros((self.nH,self.nH, self.nd)) #Initialize state-transition matrix
        self.Htrans[:, :, 0] = np.identity(self.nH) # Identity matrix if choosing unemployment
        self.Htrans[:, :, 2] = np.identity(self.nH) # Identity matrix if choosing labor
        self.Htrans[:, :, 1] = np.eye(self.nH, k=1) #Create matrix for schooling
        self.Htrans[-1, -1, 1] = 1 #Set terminal condition

        #State transition in K
        self.Ktrans = np.zeros((self.nK,self.nK, self.nd)) #Initialize state-transition matrix
        self.Ktrans[:, :, 0] = np.identity(self.nK) # Identity matrix if choosing unemployment
        self.Ktrans[:, :, 1] = np.identity(self.nK) # Identity matrix if choosing schooling
        self.Ktrans[:, :, 2] = np.eye(self.nK, k=1) #Create matrix for labor
        self.Ktrans[-1, -1, 2] = 1 #Set terminal condition

        #State transition in A
        self.Atrans = np.zeros((self.nA,self.nA, self.nd)) #Initialize state-transition matrix
        for d in range(self.nd):
            self.Atrans[:, :, d] = np.eye(self.nA, k=1) #Create deterministic state 
            self.Atrans[-1, -1, d] = 1
            
        #State transition in dlag
        self.dlagtrans = np.zeros((self.nd,self.nd, self.nd)) #Initialize state-transition matrix
        for d in range(self.nd):
            self.dlagtrans[:, d, d] = 1 #Create deterministic state 

        #State transition in G
        self.Gtrans = np.zeros((self.nG,self.nG, self.nd)) #Initialize state-transition matrix
        for d in range(self.nd):
            self.Gtrans[:, :, d] = np.eye(self.nG) #Create deterministic state 
        #Joint state transition
        self.nX, self.Xtrans, = self._tensorproduct(self.Htrans,self.Ktrans, self.Atrans, self.dlagtrans, self.Gtrans)

    def util_CRRA(self, c):
        #CRRA utility
        return c**(1-self.gamma)/(1-self.gamma)
    
    def Util(self, output = 1):
        ##### Compute utility of choices - also derivate of utility
        Utility = np.zeros((self.nX, self.nd)) # Initialize array

        Utility[:, 0] = self.util_CRRA(self.wage(self.Hgrid, self.Kgrid, self.Agrid,self.dgrid, self.Ggrid , 0)) #Utility of unemployment
        Utility[:, 1] = self.util_CRRA(self.wage(self.Hgrid, self.Kgrid, self.Agrid,self.dgrid, self.Ggrid , 1)) # Utility of schooling 
        Utility[:, 2] = self.util_CRRA(self.wage(self.Hgrid, self.Kgrid, self.Agrid,self.dgrid, self.Ggrid , 2)) # Utility of labor

        if output == 1:
            return Utility
        
        #Compute Utility derivatives - done in the linear case
        w, dw = self.wage(H=self.Hgrid,K=self.Kgrid, A=self.Agrid, dlag=self.dgrid, G = self.Ggrid ,d=None,output=2)

        dU = dw
        for i_param in range(len(self.pnames)):
            for i_d in range(self.nd):
                dU[:,i_param,i_d] *= w[:,i_d]**(-self.gamma)
        if output==2:
            return Utility, dU


    def wage(self, H, K, A, dlag, G, d, output=1):
        ### Function used for utility and wages 
        #H, K, A and so on are grids
        #### We tile and repeat everywhere to get the dimensions of X right
        if output == 1:
            if d == 0:
                unempl_transfer = np.repeat(self.delta_0,self.nX) \
                                + self.kappa_G1 * np.repeat(np.repeat(np.repeat(np.repeat(np.where(G==1,1,0),self.nH),self.nK),self.nA),self.nd) \
                                + self.kappa_G2 * np.repeat(np.repeat(np.repeat(np.repeat(np.where(G==2,1,0),self.nH),self.nK),self.nA),self.nd) \
                                + self.kappa_d0 * np.tile(np.repeat(np.repeat(np.repeat(np.where(dlag == 0,1,0),self.nH),self.nK), self.nA),self.nG) \
                                - self.kappa_H1 * np.tile(np.tile(np.tile(np.tile(np.where(H >= 3,1,0),self.nK),self.nA), self.nd), self.nG)  


                                
                return unempl_transfer
            if d==1:
                stud_transfer = np.repeat(self.delta_1,self.nX) \
                            + self.phi_G1 * np.repeat(np.repeat(np.repeat(np.repeat(np.where(G==1,1,0),self.nH),self.nK),self.nA),self.nd) \
                            + self.phi_G2 * np.repeat(np.repeat(np.repeat(np.repeat(np.where(G==2,1,0),self.nH),self.nK),self.nA),self.nd) \
                            - self.phi_H1 * np.tile(np.tile(np.tile(np.tile(np.where(H >= 3,1,0),self.nK),self.nA), self.nd), self.nG) \
                            - self.phi_H2 * np.tile(np.tile(np.tile(np.tile(np.where(H >= 6,1,0),self.nK),self.nA), self.nd), self.nG) \
                            - self.phi_H3 * np.tile(np.tile(np.tile(np.tile(np.where(H >= 9,1,0),self.nK),self.nA), self.nd), self.nG) \
                            + self.phi_d0 * np.tile(np.repeat(np.repeat(np.repeat(np.where(dlag == 1,1,0),self.nH),self.nK), self.nA),self.nG) \
                            + self.A_dummy * np.tile(np.tile(np.repeat(np.repeat(np.where(A<=20, 1, 0), self.nH),self.nK), self.nd), self.nG) 

 
                            
                return stud_transfer
            if d==2:
                logwage = np.repeat(self.delta_2,self.nX) \
                    + self.alpha_H1 * np.tile(np.tile(np.tile(np.tile(H,self.nK),self.nA), self.nd), self.nG) \
                    + self.alpha_H2 * np.tile(np.tile(np.tile(np.tile(H**2,self.nK),self.nA), self.nd), self.nG) \
                    + self.alpha_K1 * np.tile(np.tile(np.tile(np.repeat(K, self.nH),self.nA), self.nd), self.nG) \
                    + self.alpha_K2 * np.tile(np.tile(np.tile(np.repeat(K**2,self.nH),self.nA), self.nd), self.nG) \
                    + self.alpha_A1 * np.tile(np.tile(np.repeat(np.repeat(A, self.nH),self.nK), self.nd), self.nG) \
                    + self.alpha_A2 * np.tile(np.tile(np.repeat(np.repeat(A**2, self.nH),self.nK), self.nd), self.nG) \
                    + self.alpha_G1 * np.repeat(np.repeat(np.repeat(np.repeat(np.where(G == 1,1,0),self.nH),self.nK),self.nA),self.nd) \
                    + self.alpha_G2 * np.repeat(np.repeat(np.repeat(np.repeat(np.where(G == 2,1,0),self.nH),self.nK),self.nA),self.nd)
                return self.r *  np.exp(logwage)
            else:
                raise ValueError("not yet implemented")

        if output == 2:
            ### Wage in matrix form
            w = np.zeros([self.nX, self.nd])
            #Unemployment
            w[:,0]  = np.repeat(self.delta_0,self.nX) \
                    + self.kappa_G1 * np.repeat(np.repeat(np.repeat(np.repeat(np.where(G==1,1,0),self.nH),self.nK),self.nA),self.nd) \
                    + self.kappa_G2 * np.repeat(np.repeat(np.repeat(np.repeat(np.where(G==2,1,0),self.nH),self.nK),self.nA),self.nd) \
                    + self.kappa_d0 * np.tile(np.repeat(np.repeat(np.repeat(np.where(dlag == 0,1,0),self.nH),self.nK), self.nA),self.nG) \
                    - self.kappa_H1 * np.tile(np.tile(np.tile(np.tile(np.where(H >= 3,1,0),self.nK),self.nA), self.nd), self.nG) 

            #Students
            w[:,1] = np.repeat(self.delta_1,self.nX) \
                            + self.phi_G1 * np.repeat(np.repeat(np.repeat(np.repeat(np.where(G==1,1,0),self.nH),self.nK),self.nA),self.nd) \
                            + self.phi_G2 * np.repeat(np.repeat(np.repeat(np.repeat(np.where(G==2,1,0),self.nH),self.nK),self.nA),self.nd) \
                            - self.phi_H1 * np.tile(np.tile(np.tile(np.tile(np.where(H >= 3,1,0),self.nK),self.nA), self.nd), self.nG) \
                            - self.phi_H2 * np.tile(np.tile(np.tile(np.tile(np.where(H >= 6,1,0),self.nK),self.nA), self.nd), self.nG) \
                            - self.phi_H3 * np.tile(np.tile(np.tile(np.tile(np.where(H >= 9,1,0),self.nK),self.nA), self.nd), self.nG)  \
                            + self.phi_d0 * np.tile(np.repeat(np.repeat(np.repeat(np.where(dlag == 1,1,0),self.nH),self.nK), self.nA),self.nG) \
                            + self.A_dummy * np.tile(np.tile(np.repeat(np.repeat(np.where(A<=20, 1, 0), self.nH),self.nK), self.nd), self.nG) 


            w[:,2] = self.r \
                   * np.exp(np.repeat(self.delta_2,self.nX) \
                   + self.alpha_H1 * np.tile(np.tile(np.tile(np.tile(H,self.nK),self.nA), self.nd), self.nG) \
                   + self.alpha_H2 * np.tile(np.tile(np.tile(np.tile(H**2,self.nK),self.nA), self.nd), self.nG) \
                   + self.alpha_K1 * np.tile(np.tile(np.tile(np.repeat(K, self.nH),self.nA), self.nd), self.nG) \
                   + self.alpha_K2 * np.tile(np.tile(np.tile(np.repeat(K**2,self.nH),self.nA), self.nd), self.nG) \
                   + self.alpha_A1 * np.tile(np.tile(np.repeat(np.repeat(A, self.nH),self.nK), self.nd), self.nG) \
                   + self.alpha_A2 * np.tile(np.tile(np.repeat(np.repeat(A**2, self.nH),self.nK), self.nd), self.nG) \
                   + self.alpha_G1 * np.repeat(np.repeat(np.repeat(np.repeat(np.where(G == 1,1,0),self.nH),self.nK),self.nA),self.nd) \
                   + self.alpha_G2 * np.repeat(np.repeat(np.repeat(np.repeat(np.where(G == 2,1,0),self.nH),self.nK),self.nA),self.nd))

            ### Derivative in matrix form
            dw = np.zeros([self.nX,len(self.pnames), self.nd])
            
            for i_param in range(len(self.pnames)):
                #Unemployed
                if self.pnames[i_param] == "delta_0":
                    dw[:,i_param,0] = 1
                if self.pnames[i_param] == "kappa_G1":
                    dw[:,i_param,0] = np.repeat(np.repeat(np.repeat(np.repeat(np.where(G == 1,1,0), self.nH),self.nK), self.nd),self.nA)
                if self.pnames[i_param] == "kappa_G2":
                    dw[:,i_param,0] = np.repeat(np.repeat(np.repeat(np.repeat(np.where(G == 2,1,0), self.nH),self.nK), self.nd),self.nA)
                if self.pnames[i_param] == "kappa_d0":
                    dw[:,i_param,0] = np.tile(np.repeat(np.repeat(np.repeat(np.where(dlag == 0,1,0),self.nH),self.nK), self.nA),self.nG)
                if self.pnames[i_param] == "kappa_H1":
                    dw[:,i_param,0] = - np.tile(np.tile(np.tile(np.tile(np.where(H >=3,1,0),self.nK),self.nA), self.nd),self.nG)
                #Students
                if self.pnames[i_param] == "delta_1":
                    dw[:,i_param,1] = 1
                if self.pnames[i_param] == "phi_G1":
                    dw[:,i_param,1] = np.repeat(np.repeat(np.repeat(np.repeat(np.where(G == 1,1,0), self.nH),self.nK), self.nd),self.nA)
                if self.pnames[i_param] == "phi_G2":
                    dw[:,i_param,1] = np.repeat(np.repeat(np.repeat(np.repeat(np.where(G == 2,1,0), self.nH),self.nK), self.nd),self.nA)
                if self.pnames[i_param] == "phi_H1":
                    dw[:,i_param,1] = - np.tile(np.tile(np.tile(np.tile(np.where(H >=3,1,0),self.nK),self.nA), self.nd),self.nG)
                if self.pnames[i_param] == "phi_H2":
                    dw[:,i_param,1] = - np.tile(np.tile(np.tile(np.tile(np.where(H >=6,1,0),self.nK),self.nA), self.nd),self.nG)
                if self.pnames[i_param] == "phi_H3":
                    dw[:,i_param,1] = - np.tile(np.tile(np.tile(np.tile(np.where(H >=9,1,0),self.nK),self.nA), self.nd),self.nG)
                if self.pnames[i_param] == "phi_d0":
                    dw[:,i_param,1] = np.tile(np.repeat(np.repeat(np.repeat(np.where(dlag == 1,1,0),self.nH),self.nK), self.nA),self.nG)
                if self.pnames[i_param] == "A_dummy":
                    dw[:,i_param,1] = np.tile(np.tile(np.repeat(np.repeat(np.where(A<=20, 1, 0), self.nH),self.nK), self.nd), self.nG)
                #Workers
                if self.pnames[i_param] == "delta_2":
                    dw[:,i_param,2] = w[:,2]
                if self.pnames[i_param] == "alpha_H1":
                    dw[:,i_param,2] = np.tile(np.tile(np.tile(np.tile(H,self.nK),self.nA), self.nd),self.nG) * w[:,2]
                if self.pnames[i_param] == "alpha_H2":
                    dw[:,i_param,2] = np.tile(np.tile(np.tile(np.tile(H**2,self.nK),self.nA), self.nd),self.nG) * w[:,2]
                if self.pnames[i_param] == "alpha_K1":
                    dw[:,i_param,2] = np.tile(np.tile(np.tile(np.repeat(K, self.nH),self.nA), self.nd),self.nG) * w[:,2]
                if self.pnames[i_param] == "alpha_K2":
                    dw[:,i_param,2] = np.tile(np.tile(np.tile(np.repeat(K**2,self.nH),self.nA), self.nd),self.nG) * w[:,2]
                if self.pnames[i_param] == "alpha_A1":
                    dw[:,i_param,2] = np.tile(np.tile(np.repeat(np.repeat(A, self.nH),self.nK), self.nd),self.nG) * w[:,2]
                if self.pnames[i_param] == "alpha_A2":
                    dw[:,i_param,2] = np.tile(np.tile(np.repeat(np.repeat(A**2, self.nH),self.nK), self.nd),self.nG) * w[:,2]
                if self.pnames[i_param] == "alpha_G1":
                    dw[:,i_param,2] = np.repeat(np.repeat(np.repeat(np.repeat(np.where(G == 1,1,0), self.nH),self.nK), self.nd),self.nA) * w[:,2]
                if self.pnames[i_param] == "alpha_G2":
                    dw[:,i_param,2] = np.repeat(np.repeat(np.repeat(np.repeat(np.where(G == 2,1,0), self.nH),self.nK), self.nd),self.nA) * w[:,2]
            return w, dw



    def bellman(self, EV0, output=1):
        #### Evaluate bellmann operator
        #Dimension of value is entirety choice space and everything
        #except last element in state-space for H since final period needs to be handled elsewhere
        value = np.zeros((self.nX, self.nd))
        EV1 = np.zeros((self.nX))
        CCP = np.zeros((self.nX, self.nd))
        dV = np.zeros((self.nX, self.nX))

        Utility = self.Util()

        # For each choice 
        for d in self.dgrid:
            # Compute choice-specific value function        
            value[:, d] =  Utility[:, d] + self.beta * self.Xtrans[d] @ EV0[:]
        #Compute re-centered logsum
        maxV = np.amax(value, axis=1)
        maxV = np.reshape(maxV, (self.nX,1))

        logsum = maxV[:, 0] + self.eps_sigma * np.log(np.sum(np.exp((value - maxV)/self.eps_sigma), axis=1))

        EV1 = logsum # Bellman operator is the integrated value function

        for d in self.dgrid:
            #Compute conditional choice probability
            CCP[:, d] = np.exp((value[:, d] - logsum)/self.eps_sigma)
            #Compute derivate of Bellman equation
            if output == 2:
                dV +=  self.Xtrans[d].multiply(self.beta * CCP[:,d]).toarray()
                
        if output == 1:
            return EV1, CCP
        if output == 2:
            return EV1, CCP, dV

    def predicted_wage(self,H,K,A,G):
        #### Compute predicted wage
        wage = self.r \
             * np.exp((self.delta_2 \
             + self.alpha_H1 * H \
             + self.alpha_H2 * H**2 \
             + self.alpha_K1 * K \
             + self.alpha_K2 * K**2 \
             + self.alpha_A1 * A \
             + self.alpha_A2 * A**2 \
             + self.alpha_G1 * np.where(G==1,1,0) \
             + self.alpha_G2 * np.where(G==2,1,0)))
        
        return wage


    def simulate(self, CCP, data=None):
        ####### Simulate Data

        np.random.seed(2020)


        if data is None:
            #Initialize state variable
            self.sim.H[0, :] = 0
            self.sim.K[0, :] = 0
            self.sim.A[0, :] = self.A_min
            self.sim.dlag[0, :] = 2
            self.sim.G[0, :] = np.random.choice(self.nG, size=self.N)
        else:
            initial_state = data.loc[pd.IndexSlice[:,:,18] ][["H", "K", "G", "dlag"]]
            self.N = initial_state.shape[0]
            self.create_grids()
            #Initialize state variable
            self.sim.H[0, :] = initial_state["H"]
            self.sim.K[0, :] = initial_state["K"]
            self.sim.A[0, :] = self.A_min
            self.sim.dlag[0, :] = initial_state["dlag"]
            if np.sum(initial_state["G"].isna()) == 0:
                self.sim.G[0, :] = initial_state["G"]
            else:
                self.sim.G[0, :] = np.random.choice(self.nG, size=self.N)
        # Not dependent on data
        self.sim.d[:, :] = 0
        self.sim.X[:, :] = 0
        self.sim.w[:, :] = 0



        #Initiate random Gumbel
        # euler_constant = 0.577215664901532
        # scale = self.eps_sigma * np.sqrt(6) / np.pi
        # location = -scale * euler_constant
        scale = self.nu_sigma
        location = -self.nu_sigma**2/2
        self.sim.eps = np.random.normal(loc=location, scale=scale, size=(self.nT,self.N))

        # Drawing random variables for choices
        u_d  = np.random.rand(self.nT, self.N)
        csum_CCP = np.cumsum(CCP, axis=1)
        #Loop over individuals and time periods
        for id in range(self.N):
            for t in range(self.nT):
                #Get Index for state variables
                index =  self.sim.G[t,id] * self.nd * self.nA * self.nK * self.nH + self.sim.dlag[t,id] * self.nA * self.nK * self.nH + (self.sim.A[t, id]-self.A_min)*self.nK*self.nH + self.sim.K[t, id]*self.nH + self.sim.H[t, id]
                #Save index variable
                self.sim.X[t, id] = index
                # Compute Choices by cumulated sum of CCP
                for val in csum_CCP[index,:]:
                    self.sim.d[t, id] += u_d[t, id] > val
                #Given Choice - Handle state transitions
                if t< self.nT - 1:
                    if self.sim.d[t,id] == 0:
                        self.sim.H[t+1, id] = self.sim.H[t, id]
                        self.sim.K[t+1, id] = self.sim.K[t, id]
                        self.sim.dlag[t+1, id] = 0

                    if self.sim.d[t,id] == 1:
                        self.sim.H[t+1, id] = min(self.sim.H[t, id] + 1, self.nH-1)
                        self.sim.K[t+1, id] = self.sim.K[t, id]
                        self.sim.dlag[t+1, id] = 1

                    if self.sim.d[t,id] == 2:
                        self.sim.H[t+1, id] = self.sim.H[t, id]
                        self.sim.K[t+1, id] = min(self.sim.K[t, id] + 1, self.nK-1)
                        self.sim.dlag[t+1, id] = 2

                    #Handle deterministic state transition
                    self.sim.A[t+1, id] = np.where(self.sim.A[t, id]<self.A_max, self.sim.A[t, id] + 1, self.A_max)
                    self.sim.G[t+1, id] = self.sim.G[t, id]
                #Compute wage
                self.sim.w[t,id] = self.predicted_wage(self.sim.H[t,id], self.sim.K[t,id], self.sim.A[t,id], self.sim.G[t,id]) *  np.exp(self.sim.eps[t,id])





class maximum_likelihood_vec():
    #### Object for estimation procedure
    def __init__(self,ll_wage_scale = 1,**kwargs):
        self.ll_wage_scale = ll_wage_scale
        pass

    def updatepar(self, par, parnames, parvals):  # used to update parameter values for which LL is evaluated
        #### Updates parameters

        for i,parname in enumerate(parnames):
            parval = parvals[i]
            setattr(par,parname,parval) # It gives the attibute parname the new value parval, within the par class


    def log_likelihood_PANEL(self, data, model, theta, estpar, solver, include_wage, include_choice, algorithm):
        ##### Log-likelihood function
        
        print("Guess\n",theta)

        #Log likelihood from main model
        if include_choice:
            LLc_choice = self.log_likelihood_choice(model,data, solver, estpar, theta, algorithm=algorithm)
        else:
            LLc_choice=0

        #Log likelihood from wage estimation
        if include_wage:
            LLc_wage = self.log_likelihood_wage(model,data, estpar, theta)
        else:
            LLc_wage = 0

        #Log likelihood contribution in total
        LLc = LLc_choice + LLc_wage

        #Log likelihood
        LL = np.mean(LLc)
        if include_choice==False: #LL too small if only wage is included
            LL *= self.ll_wage_scale
        #print('LL',LL)

        return LL
        
    def log_likelihood_choice(self,model,data,solver, estpar, theta, algorithm, output=1):
        ### Log likelihood from choice

        #Update model parameters
        self.updatepar(model, estpar, theta)

        #Solve model
        V0 = np.zeros((model.nX))
        if algorithm=='poly':
            EV, CCP, dV, iter_sa, iter_nk= solver.poly(model.bellman, V0, beta = model.beta ,output = 5)
        if algorithm=='sa':
            EV, CCP, iter_sa= solver.sa(model.bellman, V0, beta = 0)
            iter_nk = 0
            dV = np.nan
        # Create dummmies for choice and stack variables correctly
        d_j = np.column_stack((np.where(data["d"]==0, 1, 0), np.where(data["d"]==1, 1, 0), np.where(data["d"]==2, 1, 0)))
        CCP_j = np.column_stack((CCP[data["X"], 0], CCP[data["X"], 1], CCP[data["X"], 2]))

        #Compute likelihood contribution
        LLc = np.log(np.sum( CCP_j * d_j, axis = 1))


        if output == 1:
            return LLc
        if output == 2:
            if algorithm == 'sa':
                EV, CCP, dV = model.bellman(EV, output=2)
            return LLc, d_j, CCP_j, EV, CCP, dV, iter_sa, iter_nk

    def log_likelihood_wage(self,model,data, estpar, theta, output=1):
        ##### Likelihood of wge

        #Update model parameters
        self.updatepar(model, estpar, theta)

        sigma = model.nu_sigma
        #Predicted log wage
        wgrid,dwgrid = model.wage(model.Hgrid,model.Kgrid, model.Agrid, model.dgrid, model.Ggrid, 2,output=2)
        w_predict = wgrid[data["X"],2]


        #PDF of extreme value distribution
        location_predict = np.log(w_predict)

        scipy_scale = np.exp(location_predict-sigma**2/2)

        # pdf = lambda x: (1 / scale) * np.exp(-(((x[0] - x[1]) / scale) + np.exp(-((x[0] - x[1]) / scale))))
        pdf = lambda x: stats.lognorm.pdf(x[0], sigma, loc=0, scale=x[1])
        #Log likelihood contribution
        likelihood = pdf((data["w"],scipy_scale))+1e-12
        LLc = np.where(data["d"]==2,np.log(likelihood), 0 )
        # LLc = np.log(likelihood)*np.where(data["d"]==2, 1, 0)

        if output == 1:
            return LLc
        if output == 2:
            return LLc, w_predict, sigma, likelihood

    def score(self, data, model, solver, estpar, theta, include_wage, include_choice):
        #### Compute Scores
        #Score from main model
        if include_choice:
            score_choice = self.score_choice_alt(data, model, solver, estpar, theta)
        else:
            score_choice = 0

        #Score from wage estimation
        if include_wage:
            score_wage = self.score_wage(data,model, estpar, theta)
        else:
            score_wage = 0

        score = score_choice + score_wage

        return score


    def score_choice(self,data, model, solver, estpar, theta):
        #Evalute log-likelihood function to get stuff we need below
        LLc, d_j, CCP_j, EV, CCP, dV0, iter_sa, iter_nk = self.log_likelihood_choice(model,data,solver, estpar, theta, algorithm='sa', output=2,)
        X = data["X"]
        #Get derivative of Utility
        U, dU = model.Util(output = 2)


        #Step 1: Derivative of operator with respect to parameters
        dbellman = 0
        for d in model.dgrid:
            dbellman += CCP[:, d][:, None] * dU[:, :, d]

        #Step 2: Compute derivate of fixed poiint with respect to parameters
        F = np.eye(model.nX) - dV0 #Identity minus derivate with respect to expectation

        dV = np.linalg.solve(F, dbellman)

        if 'beta' in estpar:
            dbellman_beta = 0
            for d in model.dgrid:
                # EV_data = EV[data["X"]]
                dbellman_beta += CCP[:, d][:, None] * model.Xtrans[:, :, d] @ EV
            dV_beta = np.linalg.solve(F, dbellman_beta)




        #Step 3: Compute derivative of log-like wirt to parameters
        score = 0
        for i in range(model.nd):
            dv = dU[:, :, i] + model.beta * model.Xtrans[i] @ dV
            if 'beta' in estpar:
                index = estpar.index('beta')
                dv[:, index] = model.Xtrans[i] @ EV + model.beta * model.Xtrans[i] @ dV_beta
            score += (d_j[:, i] - CCP_j[:, i])[:, None] * dv[X, :] / model.eps_sigma
        if 'eps_sigma' in estpar:
            index = estpar.index('eps_sigma')
            score[:, index] = self.score_choice_alt(data,model,solver,['eps_sigma'], np.array(theta[index]).reshape(-1,1)).reshape(-1,)
        return score

    def score_choice_alt(self,data, model, solver, estpar, theta):
        ### Numerical score
        LLc, d_j, CCP_j, EV, CCP, dV0, iter_sa, iter_nk = self.log_likelihood_choice(model,data,solver, estpar, theta, algorithm='sa', output=2,)

        #Evalute log-likelihood function to get stuff we need below
        like_choice_lambda = lambda x: self.log_likelihood_choice(model,data,solver,estpar, x,'sa')
        score = centered_grad(like_choice_lambda, theta)
        print(score)
        return score




    def score_wage(self,data,model, estpar, theta):
        #### Score wages
        #Evalute log-likelihood function to get stuff we need below
        LLc, w_predict, sigma, likelihood = self.log_likelihood_wage(model,data, estpar, theta, output=2)
        
        #Get derivative of Utility
        pred_w, dw = model.wage(model.Hgrid,model.Kgrid,model.Agrid, model.dgrid, model.Ggrid, None, output=2)

        #get wage from data
        w = data["w"]

        #Calculate score from parameters in mean wage
        score = (1 / sigma**2) * (np.log(w)[:,None] - np.log(w_predict)[:,None] + sigma**2 / 2) * 1 / w_predict[:,None] * dw[data['X'],:,2]
        if 'nu_sigma' in estpar:
            index = estpar.index('nu_sigma')
            # score[:, index] += -1 / scale  + (w - loc)**2 *  (1 / scale**3)
            score[:, index] += -1 / sigma + (1 / sigma**3) *  (np.log(w) - np.log(w_predict) + sigma**2 / 2)**2 - (1/sigma) * (np.log(w) - np.log(w_predict) + sigma**2 / 2)
        # print(score)
        score_workers = np.where(data['d'][:,None]==2,score,0.)
        
        return score_workers


    def score_wage_alt(self,data,model, estpar, theta):
        ### Numerical score
        #Evalute log-likelihood function to get stuff we need below
        LLc, w_predict, sigma, likelihood = self.log_likelihood_wage(model,data, estpar, theta, output=2)
        


        #get wage from data
        w = data["w"]
        like_wage_lambda = lambda x: self.log_likelihood_wage(model,data,estpar,x)
        score_workers = centered_grad(like_wage_lambda, theta)

        
        return score_workers




    def grad(self, data, model, solver, estpar, theta, include_wage, include_choice):
        #Compute log-like gradient
        score = self.score(data, model, solver, estpar, theta, include_wage=include_wage, include_choice=include_choice)
        if include_choice==False:
            score *= self.ll_wage_scale #Because we have increased LL for numeric reasons
        #print('evaluate score',np.mean(score, axis = 0))
        return - np.mean(score, axis = 0) # Mean of score

    def hes(self, data, model, solver, estpar, theta, include_wage, include_choice):
        #Compute Hessian as outer product of scores using information identity - computing hessian as variance of scores
        score = self.score(data, model, solver, estpar, theta, include_wage=include_wage, include_choice=include_choice)
        hessian = score.T @ score / data["X"].shape[0]
        if include_choice==False:
            hessian *= self.ll_wage_scale
        #print('hessian',hessian)
        return hessian
        
    def maximum_likelihood_Panel(self,theta0, model, estpar, data, bounds, solver, method2, include_wage=True, include_choice=True, only_est=False, algorithm='sa'):
        ### Function for estimation.
        #Define bounds
        bnds = bounds

        samplesize = data["X"].shape[0]
        
        model.pnames = estpar

        # Check the parameters
        assert (len(estpar)==len(theta0)), 'Number of parameters and initial values do not match'
        
        #Estimation
        obj_fun = lambda x: -self.log_likelihood_PANEL(data, model, x, estpar,solver, include_wage=include_wage, include_choice=include_choice, algorithm=algorithm) # negative as use minimiser
        jacobian_func = lambda x: self.grad(data, model, solver, estpar, x, include_wage, include_choice)
        hessian_func = lambda x: self.hes(data, model, solver, estpar, x, include_wage, include_choice)

        #Start time
        t0 = time.time()
        options={'gtol':1e-06}

        if method2 == "BNJ":
            res = optimize.minimize(obj_fun,theta0, bounds = bnds, method = "L-BFGS-B", tol=1e-7)
        elif method2 == "BJ":
            res = optimize.minimize(obj_fun,theta0, bounds = bnds, method = "L-BFGS-B", tol=1e-7, jac=jacobian_func)
        elif method2 == "NCG":
            res = optimize.minimize(obj_fun,theta0, method = 'trust-ncg',jac = jacobian_func, hess = hessian_func, tol=1e-6)
        else: 
            res = optimize.minimize(obj_fun,theta0, method = 'trust-ncg',jac = jacobian_func, hess = hessian_func, tol=1e-6, options=options)
           
        theta_result = res.x

        #Update Parameters
        self.updatepar(model,estpar,list(res.x))
        print(res)
        print(f'Estimation done in {(time.time()-t0)//60:.0f} minutes and {(time.time()-t0)%60:.1f} seconds')

        if only_est == True:
            return res, theta_result


        hessian = self.hes(data, model, solver, estpar, np.array(theta_result), include_wage, include_choice)
        Avar = np.linalg.inv(hessian * samplesize)
        stderr = np.sqrt(np.diag(Avar))

        return res, theta_result, stderr, Avar
        #return res, theta_result

def centered_grad(f, x0, h:float=1.49e-08):
    '''centered_grad: numerical gradient calculator
    Args.
        f: function handle taking *one* input, f(x0). f can return a vector. 
        x0: P-vector, the point at which to compute the numerical gradient 

    Returns
        grad: N*P matrix of numericalgradients. 
    '''
    assert x0.ndim == 1, f'Assumes x0 is a flattened array'
    P = x0.size 

    # evaluate f at baseline 
    f0 = f(x0)
    N = f0.size

    # intialize output 
    grad = np.zeros((N, P))
    for i in range(P): 

        # initialize the step vectors 
        x1 = x0.copy()  # forward point
        x_1 = x0.copy() # backwards 

        # take the step for the i'th coordinate only 
        if x0[i] != 0: 
            x1[i] = x0[i]*(1.0 + h)  
            x_1[i] = x0[i]*(1.0 - h)
        else:
            # if x0[i] == 0, we cannot compute a relative step change, 
            # so we just take an absolute step 
            x1[i] = h
            x_1[i] = -h
        
        step = x1[i] - x_1[i] # the length of the step we took 
        grad[:, i] = ((f(x1) - f(x_1))/step).flatten()

    return grad










