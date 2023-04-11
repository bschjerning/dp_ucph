# Import package
import numpy as np
import tools
from types import SimpleNamespace
import egm


class model_bufferstock():

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

        # Demograhpics
        par.T = 200 # Terminal age
        par.Tr = 200 # Retirement age, no retirement if TR=T
        par.age_min = 25 # Only relevant for figures

        # Preferences
        par.rho = 2
        par.beta = 0.96

        # Income parameters
        par.G = 1.03 # Age-invariant component of deterministic drift in income
        par.L = np.ones((par.T)) # Age-specific component of deterministic drift in income (If ones, then no life cycle)

        # Income shocks
        ## Log normal shocks
        par.sigma_xi = 0.1 # Transitory shock
        par.sigma_psi = 0.1 # Permanent shock
        ## Discrete shocks
        par.low_p = 0.005 # Probability of very low shock (Called pi in slides)
        par.low_val = 0 # Value of very small shock (Called mu in slides)

        # Saving and borrowing
        par.R = 1.04
        par.lambdaa = 0.0  # Maximum borrowing limit (Note you cannot have a variable named "lambda" in python)

        # Numerical integration and grids
        ## a_grid settings
        par.Na = 500 # number of points in grid for a
        par.a_max = 20 # maximum point in grid for a
        par.a_phi = 1.1 # Spacing in grid 
        ## Shock grid settings
        par.Neps = 8 # number of quadrature points for eps
        par.Npsi = 8 # number of quadrature points for psi

        # 6. simulation
        par.simN = 500000 # number of persons in simulation
        par.simT = 100 # number of periods in simulation
        par.sim_m_ini = 2.5 # initial m in simulation
        par.simlifecycle = 0 # = 0 simulate infinite horizon model


    def life_cycle_setup(self):
        ''' Setup for life-cycle model '''
        self.setup()

        par = self.par
        
        # Life-cycle settings
        par.T = 90-par.age_min # Terminal age (90 years)
        par.Tr = 65-par.age_min # Retirement age (65 years)
        par.simT = par.T
        par.simlifecycle = 1

        # Income drift
        par.L[0:par.Tr] = np.linspace(1,1/par.G,par.Tr) #Drift before retirement
        par.L[par.Tr-1] = 0.9 #Drop in income at retirement
        par.L[par.Tr-1:] = par.L[par.Tr-1:]/par.G #Drift after retirement

    def create_grids(self):
        par = self.par
        
        #1. Check parameters
        assert (par.rho >= 0), 'not rho > 0'
        assert (par.lambdaa >= 0), 'not lambda > 0'

        #2. Shocks
        ## Nodes and weights for quadrature
        ### Define epsilon
        eps,eps_w = tools.GaussHermite_lognorm(par.sigma_xi,par.Neps)
        ###Define psi
        par.psi,par.psi_w = tools.GaussHermite_lognorm(par.sigma_psi,par.Npsi)

        # Define xi
        ## Combine discrete (mu) and continuous (epsilon) trasitory shocks into one composite shock (xi)
        if par.low_p > 0:
            par.xi =  np.append(par.low_val+1e-8, (eps-par.low_p*par.low_val)/(1-par.low_p), axis=None) # +1e-8 makes it possible to take the log in simulation if low_val = 0
            par.xi_w = np.append(par.low_p, (1-par.low_p)*eps_w, axis=None)
        else: #If no discrete shock then xi=eps
            par.xi = eps
            par.xi_w = eps_w

        # Vectorize all
        ## Repeat and tile are used to create all combinations of shocks (like a tensor product)
        par.xi_vec = np.tile(par.xi,par.psi.size)       # Repeat entire array x times
        par.psi_vec = np.repeat(par.psi,par.xi.size)    # Repeat each element of the array x times
        par.xi_w_vec = np.tile(par.xi_w,par.psi.size)
        par.psi_w_vec = np.repeat(par.psi_w,par.xi.size)

        # Weights for each combination of shocks
        par.w = par.xi_w_vec * par.psi_w_vec
        assert (1-sum(par.w) < 1e-8), 'the weights does not sum to 1'
        par.Nshocks = par.w.size    # count number of shock nodes
        
        #3. Minimum a
        if par.lambdaa == 0: #No borrowing
            par.a_min = np.zeros([par.T,1])
        else: #Borrowing allowed
            # Allocate space
            par.a_min = np.nan + np.zeros([par.T,1])
            
            # Find smallest possible income shock
            psi_min = min(par.psi)
            xi_min = min(par.xi)
            
            # Debts has to be repaid before retirement
            for t in range(par.T-1,-1,-1):
                # Omega is maximum guarenteed repayable debt in period t
                if t >= par.Tr:
                    Omega = 0  # No debt in retirement
                else: 
                    Omega = par.R**(-1) * (Omega+xi_min) * par.G * par.L[t] * psi_min # Maximum debt in period t guarenteed to be repaid
                
                # Use binding borrowing constraint
                par.a_min[t]= -min(Omega,par.lambdaa)
        
        
        #4. End of period assets (a grid)
        par.grid_a = np.nan + np.zeros([par.T,par.Na])
        for t in range(par.T):
            par.grid_a[t,:] = tools.nonlinspace(par.a_min[t]+1e-8,par.a_max,par.Na,par.a_phi)

        # 5. Set seed
        np.random.seed(2020)

        #6.  Conditions (not used)
        par.FHW = par.G/par.R 
        par.AI = (par.R*par.beta)**(1/par.rho)
        par.GI = par.AI*sum(par.w*par.psi_vec**(-1))/par.G
        par.RI = par.AI/par.R      
        par.WRI = par.low_p**(1/par.rho)*par.AI/par.R
        par.FVA = par.beta*sum(par.w*(par.G*par.psi_vec)**(1-par.rho))


    ############
    # solve    #
    ############
    def solve(self):

        # Initialize
        sol = self.sol
        par = self.par

        shape=(par.T,par.Na+1)
        sol.m = np.nan+np.zeros(shape)
        sol.c = np.nan+np.zeros(shape)
        
        # Last period, (= consume all) 
        sol.m[par.T-1,:]=np.linspace(0,par.a_max,par.Na+1)
        sol.c[par.T-1,:]= sol.m[par.T-1,:].copy()

        # Before last period
        for t in range(par.T-2,-1,-1):
            # Solve model with EGM
            egm.EGM(sol,t,par)

            # add zero consumption
            sol.m[t,0] = par.a_min[t,0]
            sol.c[t,0] = 0

    def simulate (self):

        #Unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # Initialize
        shape = (par.simT, par.simN)
        sim.m = np.nan +np.zeros(shape)
        sim.c = np.nan +np.zeros(shape)
        sim.a = np.nan +np.zeros(shape)
        sim.p = np.nan +np.zeros(shape)
        sim.y = np.nan +np.zeros(shape)

        # Shocks
        shocki = np.random.choice(par.Nshocks,(par.T,par.simN),replace=True,p=par.w) #draw values between 0 and Nshocks-1, with probability w
        sim.psi = par.psi_vec[shocki] #Draw shocks from quadrature points
        sim.xi = par.xi_vec[shocki] #Draw shocks from quadrature points

        #check it has a mean of 1
        assert (abs(1-np.mean(sim.xi)) < 1e-4), 'The mean is not 1 in the simulation of xi'
        assert (abs(1-np.mean(sim.psi)) < 1e-4), 'The mean is not 1 in the simulation of psi'

        # Initial values
        sim.m[0,:] = par.sim_m_ini
        sim.p[0,:] = 0.0

        # Simulation 
        for t in range(par.simT):
            if par.simlifecycle == 0: #Approximate infinite horizon with behavior from first period
                sim.c[t,:] = tools.interp_linear_1d(sol.m[0,:],sol.c[0,:], sim.m[t,:])
            else:
                sim.c[t,:] = tools.interp_linear_1d(sol.m[t,:],sol.c[t,:], sim.m[t,:])
            
            sim.a[t,:] = sim.m[t,:] - sim.c[t,:]

            if t< par.simT-1:
                if t+1 > par.Tr: #after retirement
                    sim.m[t+1,:] = par.R*sim.a[t,:]/(par.G*par.L[t])+1
                    sim.p[t+1,:] = np.log(par.G)+np.log(par.L[t])+sim.p[t,:]
                    sim.y[t+1,:] = sim.p[t+1,:]
                else: #before retirement
                    sim.m[t+1,:] = par.R*sim.a[t,:]/(par.G*par.L[t]*sim.psi[t+1,:])+sim.xi[t+1,:]
                    sim.p[t+1,:] = np.log(par.G)+np.log(par.L[t])+sim.p[t,:]+np.log(sim.psi[t+1,:])
                    sim.y[t+1,:] = sim.p[t+1,:]+np.log(sim.xi[t+1,:])
        
        #Renormalize 
        sim.P = np.exp(sim.p)
        sim.Y = np.exp(sim.y)
        sim.M = sim.m*sim.P
        sim.C = sim.c*sim.P
        sim.A = sim.a*sim.P
