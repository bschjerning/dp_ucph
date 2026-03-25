# Import package
import numpy as np
import tools
from types import SimpleNamespace
import egm


class model_bufferstock():

    def __init__(self,name=None):
        """ defines default attributes """

        self.par = SimpleNamespace()
        self.sol = SimpleNamespace() 
        self.sim = SimpleNamespace()    
    
    ############
    # setup    #
    ############

    def setup(self):

        par = self.par

        # 1. demograhpics
        par.T = 200 # terminal age
        par.Tr = 200 # retirement age, no retirement if Tr=T
        par.age_min = 25 # only relevant for figures

        # 2. preferences
        par.rho = 2
        par.beta = 0.96

        # 3. income parameters
        par.G = 1.03 # age-invariant component of deterministic drift/growth in income
        par.L = np.ones((par.T)) # age-specific component of deterministic drift in income (If ones, then no life cycle)

        # 4. income shocks
        ## 4.a log-normal shocks
        par.sigma_xi = 0.1 # transitory shock
        par.sigma_psi = 0.1 # permanent shock

        ## 4.b discrete shocks
        par.low_p = 0.005 # probability of very low shock / unemployment (called pi in slides)
        par.low_val = 0 # value of very small shock / income in unemplyment (called mu in slides)

        # 5. saving and borrowing
        par.R = 1.04
        par.lambdaa = 0.0 # maximum borrowing limit (note you cannot have a variable named "lambda" in python)

        # 6. numerical integration and grids
        ## 6.a a_grid settings
        par.Na = 500 # number of points in grid for a
        par.a_max = 20 # maximum point in grid for a
        par.a_phi = 1.1 # Spacing in grid 

        ## 6.b shock grid settings
        par.Neps = 8 # number of quadrature points for eps
        par.Npsi = 8 # number of quadrature points for psi

        # 7. simulation
        par.simN = 500000 # number of persons in simulation
        par.simT = 100 # number of periods in simulation
        par.sim_m_ini = 2.5 # initial m in simulation
        par.simlifecycle = 0 # = 0 simulate infinite horizon model

        # 8. vectorization
        par.vec = False

    def life_cycle_setup(self):

        self.setup()
        par = self.par
        
        # 1. life-cycle settings
        par.T = 90-par.age_min # terminal age (90 years)
        par.Tr = 65-par.age_min # retirement age (65 years)
        par.simT = par.T
        par.simlifecycle = 1

        # 2. income drift
        par.L[0:par.Tr] = np.linspace(1,1/par.G,par.Tr) # drift before retirement
        par.L[par.Tr-1] = 0.9 # drop in income at retirement
        par.L[par.Tr-1:] = par.L[par.Tr-1:]/par.G # drift after retirement

    def create_grids(self):

        par = self.par
        
        # 1. check parameters
        assert (par.rho >= 0), 'not rho > 0'
        assert (par.lambdaa >= 0), 'not lambda > 0'

        # 2. shocks
        # 2.a nodes and weights for quadrature
        eps,eps_w = tools.GaussHermite_lognorm(par.sigma_xi,par.Neps)
        par.psi,par.psi_w = tools.GaussHermite_lognorm(par.sigma_psi,par.Npsi)

        # 2.b define xi: combine discrete (mu) and continuous (epsilon) trasitory shocks into one composite shock (xi)
        if par.low_p > 0:
            par.xi =  np.append(par.low_val+1e-8, (eps-par.low_p*par.low_val)/(1-par.low_p), axis=None) # +1e-8 makes it possible to take the log in simulation if low_val = 0
            par.xi_w = np.append(par.low_p, (1-par.low_p)*eps_w, axis=None)

        else: # if no discrete shock then xi=eps
            par.xi = eps
            par.xi_w = eps_w

        # 3. prepare vectorization
        # repeat and tile are used to create all combinations of shocks (like a tensor product)
        par.xi_vec = np.tile(par.xi,par.psi.size)       # Repeat entire array x times
        par.psi_vec = np.repeat(par.psi,par.xi.size)    # Repeat each element of the array x times
        par.xi_w_vec = np.tile(par.xi_w,par.psi.size)
        par.psi_w_vec = np.repeat(par.psi_w,par.xi.size)

        # weights for each combination of shocks
        par.w = par.xi_w_vec * par.psi_w_vec
        assert (1-sum(par.w) < 1e-8), 'the weights does not sum to 1'
        par.Nshocks = par.w.size    # count number of shock nodes
        
        # 4. borrowing constraint

        if par.lambdaa == 0: # no borrowing
            par.a_min = np.zeros([par.T,1])

        else: # borrowing allowed

            # a. allocate
            par.a_min = np.zeros([par.T,1]) + np.nan
            
            # b. find smallest possible income shocks
            psi_min = min(par.psi)
            xi_min = min(par.xi)
            
            # c. debts has to be repaid before retirement
            for t in reversed(range(par.T)): # T-1, T-2, ..., 0

                # i. Omega is maximum guarenteed repayable debt in period t
                if t >= par.Tr:
                    Omega = 0  #No debt in retirement

                else: 
                    Omega = par.R**(-1) * (Omega+xi_min) * par.G * par.L[t] * psi_min
                
                # ii. use binding borrowing constraint
                par.a_min[t]= max(-Omega,-par.lambdaa)
        
        # 5. end of period assets grid (a grid)
        par.grid_a = np.zeros([par.T, par.Na]) + np.nan
        for t in range(par.T):
            par.grid_a[t,:] = tools.nonlinspace(par.a_min[t]+1e-8,par.a_max,par.Na,par.a_phi)

        # 6. set seed
        np.random.seed(2026)

        # 7.  Conditions (not used)
        par.FHW = par.G/par.R 
        par.AI = (par.R*par.beta)**(1/par.rho)
        par.GI = par.AI*sum(par.w*par.psi_vec**(-1))/par.G
        par.RI = par.AI/par.R      
        par.WRI = par.low_p**(1/par.rho)*par.AI/par.R
        par.FVA = par.beta*sum(par.w*(par.G*par.psi_vec)**(1-par.rho))

    #########
    # solve #
    #########

    def solve(self):

        sol = self.sol
        par = self.par

        # 1. allocate
        sol.m = np.zeros((par.T,par.Na+1)) + np.nan
        sol.c = np.zeros((par.T,par.Na+1)) + np.nan
        
        # 2. last period, consume all
        sol.m[par.T-1, :] = np.linspace(0, par.a_max, par.Na+1)
        sol.c[par.T-1, :] = sol.m[par.T-1, :].copy()

        # 3. before last period
        for t in reversed(range(par.T-1)): # T-2, T-3, ..., 0

            # a. solve period with EGM
            egm.EGM(sol,t,par)

            # b. add zero consumption for constrained households
            sol.m[t,0] = par.a_min[t,0]
            sol.c[t,0] = 0

    def simulate (self):

        par = self.par
        sol = self.sol
        sim = self.sim

        # 1. allocate
        sim.m = np.zeros((par.simT, par.simN)) + np.nan
        sim.c = np.zeros((par.simT, par.simN)) + np.nan
        sim.a = np.zeros((par.simT, par.simN)) + np.nan
        sim.p = np.zeros((par.simT, par.simN)) + np.nan
        sim.y = np.zeros((par.simT, par.simN)) + np.nan

        # 2. shocks
        shocki = np.random.choice(par.Nshocks,(par.T, par.simN),replace=True,p=par.w) # draw values between 0 and Nshocks-1, with probability w
        sim.psi = par.psi_vec[shocki] # draw shocks from quadrature points
        sim.xi = par.xi_vec[shocki] # draw shocks from quadrature points

        # 3. check shocks have mean 1
        assert (abs(1-np.mean(sim.xi)) < 1e-3), 'The mean is not 1 in the simulation of xi'
        assert (abs(1-np.mean(sim.psi)) < 1e-3), 'The mean is not 1 in the simulation of psi'

        # 4. initial values
        sim.m[0, :] = par.sim_m_ini
        sim.p[0,: ] = 0.0

        # 5. simulation 
        for t in range(par.simT):

            if par.simlifecycle == 0: # approximate infinite horizon with behavior from first period
                sim.c[t, :] = tools.interp_linear_1d(sol.m[0, :], sol.c[0, :], sim.m[t, :])

            else:
                sim.c[t, :] = tools.interp_linear_1d(sol.m[t, :], sol.c[t, :], sim.m[t, :])
            
            # state transition
            sim.a[t, :] = sim.m[t, :] - sim.c[t, :]
            if t < par.simT-1:

                # after retirement
                if t+1 > par.Tr:
                    sim.m[t+1, :] = par.R*sim.a[t, :]/(par.G*par.L[t])+1
                    sim.p[t+1, :] = np.log(par.G)+np.log(par.L[t])+sim.p[t, :]
                    sim.y[t+1, :] = sim.p[t+1, :]

                # before retirement
                else:
                    sim.m[t+1, :] = par.R*sim.a[t, :]/(par.G*par.L[t]*sim.psi[t+1, :])+sim.xi[t+1, :]
                    sim.p[t+1, :] = np.log(par.G)+np.log(par.L[t])+sim.p[t, :]+np.log(sim.psi[t+1, :])
                    sim.y[t+1, :] = sim.p[t+1, :]+np.log(sim.xi[t+1, :])
        
        # 6. renormalize 
        sim.P = np.exp(sim.p)
        sim.Y = np.exp(sim.y)
        sim.M = sim.m*sim.P
        sim.C = sim.c*sim.P
        sim.A = sim.a*sim.P