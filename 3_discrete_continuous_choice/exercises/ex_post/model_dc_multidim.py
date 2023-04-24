# import packages 
import numpy as np
import tools
from types import SimpleNamespace
import egm_dc_multidim as egm


class model_dc_multidim():

    def __init__(self,name=None):
        """ defines default attributes """

        # a. name
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace() 
        self.sim = SimpleNamespace() 

    def setup(self):

        par = self.par

        par.T = 10

        # Model parameters
        par.rho = 2
        par.beta = 0.96
        par.alpha = 0.75
        par.kappa = 0.5
        par. R = 1.04
        par.W = 1
        par.sigma_xi = 0.1
        par.sigma_eta = 0.1

        # Grids and numerical integration
        par.m_max = 10
        par.m_phi = 1.1 # Curvature parameters
        par.a_max = 10
        par.a_phi = 1.1  # Curvature parameters
        par.p_max = 2.0
        par.p_phi = 1.0 # Curvature parameters

        par.Nxi = 8
        par.Nm = 150
        par.Na = 150
        par.Np = 100

        par.Nm_b = 50
        

    def create_grids(self):

        par = self.par

        # Check parameters
        assert (par.rho >= 0), 'not rho > 0'

        # Shocks
        par.xi,par.xi_w = tools.GaussHermite_lognorm(par.sigma_xi,par.Nxi)
        
        # End of period assets
        par.grid_a = tools.nonlinspace(0+1e-6,par.a_max,par.Na,par.a_phi)

        # Cash-on-hand
        par.grid_m =  np.concatenate([np.linspace(0+1e-6,1-1e-6,par.Nm_b), tools.nonlinspace(1+1e-6,par.m_max,par.Nm-par.Nm_b,par.m_phi)])    # Permanent income

        # Permanent income
        par.grid_p = tools.nonlinspace(0+1e-4,par.p_max,par.Np,par.p_phi)

        # Set seed
        np.random.seed(2020)

    def solve(self):
        """ solve model: Solve for discrete-choice specific consumption and value functions"""
        
        # Initialize
        par = self.par
        sol = self.sol

        shape=(par.T,2,par.Nm,par.Np)
        sol.m = np.nan+np.zeros(shape)
        sol.c = np.nan+np.zeros(shape)
        sol.v = np.nan+np.zeros(shape)
        
        # Last period, (= consume all) 
        for i_p in range(par.Np):
            for z_plus in range(2):
                sol.c[par.T-1,z_plus,:,i_p] = par.grid_m
                sol.v[par.T-1,z_plus,:,i_p] = egm.util(sol.c[par.T-1,z_plus,:,i_p],z_plus,par)

        # Before last period
        for t in range(par.T-2,-1,-1):

            #Choice specific function
            for i_p, p in enumerate(par.grid_p):
            
                for z_plus in range(2):

                    # Solve model with EGM
                    c,v = egm.EGM(sol,z_plus,p,t,par)
                    sol.c[t,z_plus,:,i_p] = c
                    sol.v[t,z_plus,:,i_p] = v
                
