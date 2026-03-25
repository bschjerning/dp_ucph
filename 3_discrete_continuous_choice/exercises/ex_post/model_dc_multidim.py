# import packages 
import numpy as np
import tools
from types import SimpleNamespace
import egm_dc_multidim as egm


class model_dc_multidim():

    def __init__(self,name=None):

        self.par = SimpleNamespace()
        self.sol = SimpleNamespace() 
        self.sim = SimpleNamespace() 

    def setup(self):

        par = self.par

        # 1. model parameters
        par.T = 10
        par.rho = 2
        par.beta = 0.96
        par.alpha = 0.75
        par.kappa = 0.5
        par. R = 1.04
        par.W = 1
        par.sigma_xi = 0.1
        par.sigma_eta = 0.1

        # 2. grids and numerical integration
        par.m_max = 10
        par.m_phi = 1.1 # curvature parameters
        par.a_max = 10
        par.a_phi = 1.1 
        par.p_max = 2.0
        par.p_phi = 1.0
        par.Nxi = 8
        par.Nm = 150
        par.Na = 150
        par.Np = 100
        par.Nm_b = 50
        
    def create_grids(self):

        par = self.par

        # 1. check parameters
        assert (par.rho >= 0), 'not rho > 0'

        # 2. shocks
        par.xi,par.xi_w = tools.GaussHermite_lognorm(par.sigma_xi,par.Nxi)
        
        # 3. end of period assets
        par.grid_a = tools.nonlinspace(0+1e-6,par.a_max,par.Na,par.a_phi)

        # 4. cash-on-hand
        par.grid_m =  np.concatenate([np.linspace(0+1e-6,1-1e-6,par.Nm_b), tools.nonlinspace(1+1e-6,par.m_max,par.Nm-par.Nm_b,par.m_phi)])    # Permanent income

        # 5. permanent income
        par.grid_p = tools.nonlinspace(0+1e-4,par.p_max,par.Np,par.p_phi)

        # 6. set seed
        np.random.seed(2026)

    def solve(self):
        
        par = self.par
        sol = self.sol

        # 1. allocate
        sol.m = np.zeros((par.T, 2, par.Nm, par.Np)) + np.nan
        sol.c = np.zeros((par.T, 2, par.Nm, par.Np)) + np.nan
        sol.v = np.zeros((par.T, 2, par.Nm, par.Np)) + np.nan
        
        # 2. last period, consume all
        for i_p in range(par.Np):

            for z_plus in range(2):

                sol.c[par.T-1, z_plus, :, i_p] = par.grid_m
                sol.v[par.T-1, z_plus, :, i_p] = egm.util(sol.c[par.T-1, z_plus, :, i_p], z_plus, par)

        # 3. backwards induction
        for t in reversed(range(par.T-1)):

            # choice specific function
            for i_p, p in enumerate(par.grid_p):
            
                for z_plus in range(2):

                    # solve model with EGM
                    c,v = egm.EGM(sol, z_plus, p, t, par)
                    sol.c[t, z_plus, :, i_p] = c
                    sol.v[t, z_plus, :, i_p] = v