# import packages 
import numpy as np
import tools
from types import SimpleNamespace
import egm_dc_exante as egm_dc

class model_dc():

    def __init__(self,name=None):
        """ defines default attributes """

        # a. name
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace() 
        self.sim = SimpleNamespace() 
    
    ############
    # setup    #
    ############

    def setup(self):

        par = self.par

        par.T = 20

        # Model parameters
        par.beta = 0.96
        par.rho = 2
        par.alpha = 0.75
        par.R = 1.04
        par.W = 1
        par.sigma_xi = 0.0
        par.sigma_eta = 0.2

        
        # Grids and numerical integration
        par.a_max = 10
        par.a_phi = 1.1 # Curvature parameters
        par.Nxi = 1     
        par.Na = 150
        par.N_bottom = 10 # Number of points at the bottom in the G2-EGM algorithm
        

    def create_grids(self):

        par = self.par

        # Check parameters
        assert (par.rho >= 0), 'not rho > 0'

        # Shocks
        par.xi,par.xi_w = tools.GaussHermite_lognorm(par.sigma_xi,par.Nxi)
        
        # End of period assets
        par.grid_a = np.nan + np.zeros([par.T,par.Na])
        for t in range(par.T):
            par.grid_a[t,:] = tools.nonlinspace(0+1e-8,par.a_max,par.Na,par.a_phi)

        # Set seed
        np.random.seed(2020)


    def solve(self):
        
        # Initialize
        par = self.par
        sol = self.sol

        shape=(par.T,2,par.Na+par.N_bottom)
        sol.m = np.nan+np.zeros(shape)
        sol.c = np.nan+np.zeros(shape)
        sol.v = np.nan+np.zeros(shape)
        
        # Last period, (= consume all) 
        for z_plus in range(2):
            sol.m[par.T-1,z_plus,:] = np.linspace(0+1e-8,par.a_max,par.Na+par.N_bottom)
            sol.c[par.T-1,z_plus,:] = np.linspace(0+1e-8,par.a_max,par.Na+par.N_bottom)
            sol.v[par.T-1,z_plus,:] = egm_dc.util(sol.c[par.T-1,z_plus,:],z_plus,par)

        # Before last period
        for t in range(par.T-2,-1,-1):

            #Choice specific fundtion
            for z_plus in range(2):

                # Solve model with EGM
                m,c,v = egm_dc.EGM(sol,z_plus,t,par)   

                # Add points at the constraints
                m_con = np.linspace(0+1e-8,m[0]-1e-8,par.N_bottom)
                c_con = m_con.copy()
                v_con = egm_dc.value_of_choice(m_con,c_con,z_plus,t,sol,par)

                sol.m[t,z_plus] = np.append(m_con, m)
                sol.c[t,z_plus] = np.append(c_con, c)
                sol.v[t,z_plus] = np.append(v_con, v)