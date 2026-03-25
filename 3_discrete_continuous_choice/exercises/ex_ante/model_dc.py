# import packages 
import numpy as np
import tools
from types import SimpleNamespace
import egm_dc

class model_dc():

    def __init__(self,name=None):

        self.par = SimpleNamespace()
        self.sol = SimpleNamespace() 
        self.sim = SimpleNamespace() 
    
    #########
    # setup #
    #########

    def setup(self):

        par = self.par

        # 1. model parameters
        par.T = 20
        par.beta = 0.96
        par.rho = 2
        par.alpha = 0.75
        par.R = 1.04
        par.W = 1
        par.sigma_xi = 0.0
        par.sigma_eta = 0.2
        
        # 2. grids and numerical integration
        par.a_max = 10
        par.a_phi = 1.1 # curvature parameters for nonlinspace
        par.Nxi = 1     
        par.Na = 150
        par.N_bottom = 10 # number of points at the bottom in the EGM algorithm
        

    def create_grids(self):

        par = self.par

        # 1. check parameters
        assert (par.rho >= 0), 'not rho > 0'

        # 2. shocks
        par.xi,par.xi_w = tools.GaussHermite_lognorm(par.sigma_xi,par.Nxi)
        
        # 3. end of period assets
        par.grid_a = tools.nonlinspace(0+1e-8,par.a_max,par.Na,par.a_phi)

        # 4. set seed
        np.random.seed(2026)


    def solve(self):

        par = self.par
        sol = self.sol

        # 1. allocate
        sol.m = np.zeros((par.T, 2, par.Na + par.N_bottom)) + np.nan
        sol.c = np.zeros((par.T, 2, par.Na + par.N_bottom)) + np.nan
        sol.v = np.zeros((par.T, 2, par.Na + par.N_bottom)) + np.nan # (period, choice, state)
        
        # 2. last period, consume all
        for z_plus in range(2):
            sol.m[par.T-1, z_plus, :] = np.linspace(0+1e-8, par.a_max, par.Na + par.N_bottom)
            sol.c[par.T-1, z_plus, :] = np.linspace(0+1e-8, par.a_max, par.Na + par.N_bottom)
            sol.v[par.T-1, z_plus, :] = egm_dc.util(sol.c[par.T-1, z_plus, :], z_plus, par)

        # 3. backwards induction
        for t in reversed(range(par.T-1)): # T-2, T-3, ..., 0

            # choice specific value functions
            for z_plus in range(2):

                # EGM-step
                m,c,v = egm_dc.EGM(sol,z_plus,t,par)   

                # Add points at the constraints - we add points at the bottom to better approximate the curvature in the value function
                # Consume everything in credit constrained region
                m_con = np.linspace(0+1e-8,m[0]-1e-8,par.N_bottom)
                c_con = m_con.copy()
                
                # find choice specific value functions
                if z_plus == 0:

                    v_con = egm_dc.value_of_choice_worker(m_con, c_con, t, sol, par)

                else:
                    v_con = egm_dc.value_of_choice_retired(m_con, c_con, t, sol, par)

                sol.m[t, z_plus] = np.append(m_con, m)
                sol.c[t, z_plus] = np.append(c_con, c)
                sol.v[t, z_plus] = np.append(v_con, v)