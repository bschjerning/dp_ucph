import torch
import numpy as np
from EconModel import EconModelClass,jit
import neural_nets
from types import SimpleNamespace
from consav.quadrature import log_normal_gauss_hermite
from consav import linear_interp
from copy import deepcopy
import numba as nb

class BufferstockClass(EconModelClass):
    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','sim', 'egm']

        # b. 
        self.nn = SimpleNamespace()
        self.other_attrs = ['nn']
    
    def setup(self):
        """ setup model """

        par = self.par
        sim = self.sim
        nn = self.nn
        egm = self.egm

        # preferences
        par.beta = 1/1.03 # discount factor
        par.rho = 2.0 # risk aversion
        par.T = 30 # number of periods

        # income process
        par.R = 1.03 # return factor
        par.sigma_psi = 0.20 # permanent income shock
        par.sigma_xi = 0.15 # transitory income shock
        par.rho_xi = 0.95 # persistence of permanent income shock

        # initial state
        par.sigma_m0 = 0.1 # std of initial cash-on-hand
        par.sigma_p0 = 0.1 # std of initial permanent income
        par.mu_p0 = 1.0 # mean of initial permanent income
        par.mu_m0 = 10.0 # mean of initial cash-on-hand

        par.Nstates = 2 # number of states
        par.Nstates_t = par.T # number of time dummies in neural net inputs. Should be equal to number of periods in model
        par.Nactions = 1 # number of actions
        par.Nshocks = 2 # number of shocks

        par.kappa = 1.0 # income coeff
        
        # sim
        sim.N = 10000 # number of agents
        sim.seed = 1928 # seed for simulation

        # nn
        nn.hidden_nodes = np.array([150,150]) # number of hidden nodes - two hidden layers here
        nn.final_activation = "sigmoid" # sets output range to [0,1] by using sigmoid activation ==> policy is predicting savings rate and not consumption directly
        nn.N = 3000 # we might want to simulate a different number of agents than we train on
        # nn.N = 1 # we might want to simulate a different number of agents than we train on
        nn.seed = 1234
        nn.K = 1000 # number of iterations
        nn.learning_rate = 1e-3
        nn.eval_freq = 10

        # egm
        egm.Nm = 150
        egm.Na = 150
        egm.Np = 100
        egm.m_max = 20
        egm.a_max = 20
        egm.m_min = 1e-8
        egm.a_min = 0
        egm.p_min = 1e-8
        egm.p_max = 10
        egm.Nxi = 8
        egm.Npsi = 8


    
    def create_neural_nets(self):
        """ create neural networks """

        nn = self.nn
        par = self.par

        # policy - output 
        nn.policy = neural_nets.Policy(par.Nstates+par.Nstates_t, # input dim = state dim
                                       par.Nactions, # output dim = action dim
                                       Nneurons=nn.hidden_nodes, # hidden nodes
                                       final_activation=nn.final_activation) # final activation - determines output range here [0,1]¨
    def create_optimizer(self):
        """ create optimizer """

        nn = self.nn

        # optimizer
        nn.policy_opt = torch.optim.Adam(nn.policy.parameters(),lr=nn.learning_rate)


    def create_grids(self):
        """ create grids for EGM """

        par = self.par
        egm = self.egm

        # a. cash-on-hand grid
        egm.m_grid = np.linspace(egm.m_min,egm.m_max,egm.Nm)

        # b. permanent income grid
        egm.p_grid = np.linspace(egm.p_min,egm.p_max,egm.Np)

        # c. action grid
        egm.a_grid = np.linspace(egm.a_min,egm.a_max,egm.Na)

        # d. consumption grid
        egm.sol_c = np.zeros((par.T,egm.Nm,egm.Np))
        egm.sol_q = np.zeros((par.T,egm.Na,egm.Np))

        egm.psi, egm.psi_w = log_normal_gauss_hermite(par.sigma_psi, egm.Npsi)
        egm.xi, egm.xi_w = log_normal_gauss_hermite(par.sigma_xi, egm.Nxi)


    def allocate(self):
        """ allocate variables """

        par = self.par
        sim = self.sim
        nn = self.nn

        # states
        sim.states = np.zeros((par.T,sim.N,par.Nstates))

        # actions
        sim.actions = np.zeros((par.T,sim.N,par.Nactions))

        # shocks
        sim.shocks = np.zeros((par.T,sim.N,par.Nshocks))

        # consumption
        sim.con = np.zeros((par.T,sim.N))


        # 2. nn
        self.create_neural_nets()
        self.create_optimizer()
        self.policy_loss_list = []




        self.create_grids()

    
    def draw_initial_states(self, N, seed=None):
        """ draw initial states """

        par = self.par

        if seed is not None:
            np.random.seed(seed)


        # a. initialize
        initial_states = np.zeros((N,par.Nstates))

        # a. initial cash-on-hand
        m0 = par.mu_m0 * np.exp(np.random.normal(-0.5*par.sigma_m0**2,par.sigma_m0,size=N))

        # b. initial permanent income
        p0 = par.mu_p0 * np.exp(np.random.normal(-0.5*par.sigma_p0**2,par.sigma_p0,size=N))

        # c. store
        initial_states[:,0] = m0
        initial_states[:,1] = p0


        return initial_states
    
    def draw_initial_shocks(self, N, T, seed=None):
        """ draw initial shocks """

        par = self.par

        if seed is not None:
            np.random.seed(seed)

        # a. initialize
        shocks = np.zeros((T,N,par.Nshocks))

        # b. draw
        shocks[:,:,0] = np.exp(np.random.normal(-0.5*par.sigma_psi**2,par.sigma_psi,(T,N)))
        shocks[:,:,1] = np.exp(np.random.normal(-0.5*par.sigma_xi**2,par.sigma_xi,(T,N)))

        return shocks
    
    def simulate(self):
        """ simulate model """

        par = self.par
        sim = self.sim
        nn = self.nn
        policy_NN = nn.policy

        states = sim.states
        actions = sim.actions
        shocks = sim.shocks

        m = states[:,:,0]
        p = states[:,:,1]
        xi = shocks[:,:,1]
        psi = shocks[:,:,0]
        con = sim.con

        
        discounted_util = np.zeros(sim.N)

        # b. simulate
        for t in range(par.T):

            # i. compute actions

            with torch.no_grad(): # no need to compute gradients
                # add time dummies
                time_dummies = np.zeros((sim.N,par.Nstates_t))
                time_dummies[:,t] = 1.0
                # convert state to pytorch tensor
                state_tensor = torch.tensor(states[t],dtype=torch.float32)
                state_tensor = torch.cat((state_tensor,torch.tensor(time_dummies,dtype=torch.float32)),dim=1)
                actions_ = policy_NN(state_tensor).clamp(0.0, 0.999).numpy() # compute savings rante and convert back to numpy
            actions[t] = actions_

            # ii. consumption
            con[t] = (1-actions[t,:,0])*m[t] # compute consumption from savings rate
            discounted_util += par.beta**t*self.utility(con[t]) # add to discounted utility

            # iii. future states
            if t < par.T-1:
                
                # p-state
                p[t+1] = p[t]**par.rho_xi * xi[t+1]
                
                # m-state
                m[t+1] = par.R*m[t]*actions[t,:,0] + p[t+1]*psi[t+1] * par.kappa
            
        # average discounted utility
        avg_discounted_util = np.mean(discounted_util)

        return avg_discounted_util

    def set_initial_state_sim(self):
        """ set initial state for simulation """

        sim = self.sim
        par = self.par

        # a. draw initial states
        sim.states[0,:,:] = self.draw_initial_states(sim.N, seed=sim.seed)

        # b. draw initial shocks
        sim.shocks = self.draw_initial_shocks(sim.N,par.T, seed=sim.seed)


    def utility(self, c):
        """ utility function """

        par = self.par

        if par.rho == 1.0:
            if type(c) == torch.Tensor:
                return torch.log(c)
            else:
                return np.log(c)
        else:
            return c**(1-par.rho)/(1-par.rho)


    def simulate_loss(self, initial_state_, shocks_):
        """ simulate model """

        par = self.par
        nn = self.nn
        policy_NN = nn.policy

        # initialize state tensors
        initial_state_ = torch.tensor(initial_state_,dtype=torch.float32)
        time_dummies = np.zeros((nn.N,par.Nstates_t))
        time_dummies[:,0] = 1.0
        initial_state = torch.cat((initial_state_,torch.tensor(time_dummies,dtype=torch.float32)),dim=1)
        state_next = torch.zeros_like(initial_state)

        # convert shocks to tensor
        shocks = torch.tensor(shocks_,dtype=torch.float32)

        # create consumption tensor

        # utility
        discounted_util = torch.zeros(nn.N)

        # b. simulate
        for t in range(par.T):

            if t == 0:
                # use initial state
                state = initial_state
            else:
                # use next state
                state = state_next.clone()

            # i. compute actions = savings-rate
            actions = policy_NN(state).clamp(0.0, 0.9999)

            # ii. consumption
            con = (1-actions[:,0])*state[:,0]

            # iii. utility
            discounted_util += par.beta**t*self.utility(con)

            # iii. future states
            if t < par.T-1:
                
                # p-state
                state_next[:,1] = state[:,1]**par.rho_xi * shocks[t+1,:,1]
                
                # m-state
                state_next[:,0] = par.R*state[:,0]*actions[:,0] + state_next[:,1]*shocks[t+1,:,0] * par.kappa

                # time dummies
                state_next[:,2+t] = 0.0
                state_next[:,2+t+1] = 1.0
        


        # d. compute loss
        loss = -torch.mean(discounted_util)

        return loss
    
    def update_policy(self, states, shocks):
        """ update policy """

        nn = self.nn
        policy_NN = nn.policy
        policy_opt = nn.policy_opt

        # a. compute loss
        policy_loss = self.simulate_loss(states, shocks)

        # b. update
        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        return policy_loss
    
    def solve(self):
        """ solve model """

        par = self.par
        sim = self.sim
        nn = self.nn


        best_avg_discounted_util = -np.inf
        best_policy = None
        np.random.seed(nn.seed)

        for k in range(nn.K):

            # i. draw states and shocks
            initial_state = self.draw_initial_states(nn.N)
            shocks = self.draw_initial_shocks(nn.N,par.T)

            # ii. update policy
            policy_loss = self.update_policy(initial_state, shocks)

            self.policy_loss_list.append(policy_loss.item())

            # iii. print
            if k % nn.eval_freq == 0:
                avg_disc_util = self.simulate()
                if avg_disc_util > best_avg_discounted_util:
                    best_policy = deepcopy(nn.policy)
                    best_avg_discounted_util = avg_disc_util
                print(f'iter: {k}, policy loss: {policy_loss.item()}, avg discounted utility: {avg_disc_util}, best avg discounted utility: {best_avg_discounted_util}')
        
        nn.policy = best_policy
        self.create_optimizer() # create new optimizer for the best policy as reference to params have been broken
        best_util = self.simulate()
        print(f'best avg discounted utility: {best_util}')
    
    def backward_induct(self):
        """ time iteration """

        par = self.par
        sim = self.sim
        nn = self.nn
        egm = self.egm


        for t in reversed(range(par.T)):
            print(f'time: {t}')

            if t == par.T-1:
                # consume all
                egm.sol_c[t] = egm.m_grid[:,np.newaxis]
            else:
                # EGM step
                self.EGM(t)
    

    
    def EGM(self,t):
        """ solve model using endogenous grid method """

        par = self.par
        sim = self.sim
        nn = self.nn
        egm = self.egm

        with jit(self) as Model:
            par = Model.par
            egm = Model.egm
            
            # 1. compute post-decision state
            postdec(par,egm,t)

            # 2. compute consumption from post-dec state and interpolate on exogenous grid
            interp_to_common_grid(par,egm,t)

    def simulate_EGM(self):
        """ simulate model with EGM-policy """

        par = self.par
        sim = self.sim
        nn = self.nn
        policy_NN = nn.policy
        egm = self.egm

        states = sim.states
        actions = sim.actions
        shocks = sim.shocks

        m = states[:,:,0]
        p = states[:,:,1]
        xi = shocks[:,:,1]
        psi = shocks[:,:,0]
        con = sim.con

        
        discounted_util = np.zeros(sim.N)


        # b. simulate
        for t in range(par.T):

            # i. consumption
            for i in range(sim.N):
                con[t,i] = linear_interp.interp_2d(egm.m_grid, egm.p_grid, egm.sol_c[t], m[t,i], p[t,i])
  
            # ii. discounted utility
            discounted_util += par.beta**t*self.utility(con[t])

            # iii. future states
            if t < par.T-1:
                
                # p-state
                p[t+1] = p[t]**par.rho_xi * xi[t+1]
                
                # m-state
                m[t+1] = par.R*(m[t]-con[t]) + p[t+1]*psi[t+1] * par.kappa
        print(f'discounted utility: {np.mean(discounted_util)}')

        


@nb.njit
def marg_util_c(par,c):
    """ marginal utility of consumption """

    return c**(-par.rho)

@nb.njit
def inv_marg_util(par, mu):
    """ inverse marginal utility """

    return mu**(-1/par.rho)


@nb.njit(parallel=False)
def postdec(par, egm, t):
    """ post decision value function """

    # loop over states
    for i_a in nb.prange(egm.Na):
        a = egm.a_grid[i_a]
        for i_p, p in enumerate(egm.p_grid):
            # loop over quadrature points
            for i_xi, xi in enumerate(egm.xi):
                for i_psi, psi in enumerate(egm.psi):
                    # p-state
                    p_plus = p**par.rho_xi*xi
                    # m-state
                    m_plus = par.R*a + p_plus*psi*par.kappa
                    # interpolate future consumption
                    c_plus = linear_interp.interp_2d(egm.m_grid, egm.p_grid, egm.sol_c[t+1], m_plus, p_plus)
                    # compute marginal utility
                    marg_util_plus = marg_util_c(par,c_plus)
                    # compute post-decision value
                    egm.sol_q[t,i_a,i_p] += marg_util_plus*egm.psi_w[i_psi]*egm.xi_w[i_xi]



@nb.njit(parallel=True)
def interp_to_common_grid(par,egm,t):
    """ interpolate to common grid """


    for i_p in nb.prange(egm.Np):

        # temporary containers
        m_temp = np.zeros((egm.Na+1))
        c_temp = np.zeros((egm.Na+1))

        for i_a in range(egm.Na):
            c_temp[i_a+1] = inv_marg_util(par, par.R*par.beta*egm.sol_q[t,i_a,i_p])
            m_temp[i_a+1] = egm.a_grid[i_a] + c_temp[i_a+1]
        
        # interpolate to common grid
        linear_interp.interp_1d_vec_mon_noprep(m_temp,c_temp,egm.m_grid, egm.sol_c[t,:,i_p])