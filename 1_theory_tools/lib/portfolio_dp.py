
import chaospy
import numpy as np
from scipy.stats import norm  
from scipy.optimize import minimize
from scipy import interpolate # Interpolation routines
import matplotlib.pyplot as plt
np.set_printoptions(precision=5, suppress=True)


# PortfolioChoiceModel class
class PortfolioChoiceModel:
    '''Dynamic portfolio choice model with log-normal income and multiple assets'''
    
    def __init__(self, beta=0.96, gamma=2, mu_y=0, sigma_y=0.1,
                 mu_R=[1.05], Sigma_R=None, rho_y=None, 
                 bnd_x=[0.1, 10], n_x=50,  n_rand=100, integration_rule="random"):

        self.beta = beta   
        self.gamma = gamma  
        self.mu_y = mu_y    
        self.sigma_y = sigma_y  

        self.N = len(mu_R)  
        self.n_choices = self.N + 1  
        self.mu_R = np.array(mu_R).reshape(self.N, 1)
        self.Sigma_R = Sigma_R if Sigma_R is not None else np.eye(self.N)  
        self.rho_y = rho_y if rho_y is not None else np.zeros((self.N,1))  

        # State space
        self.bnd_x = bnd_x  
        self.n_x = n_x  
        self.n_rand = n_rand  
        self.x0 = np.linspace(bnd_x[0], bnd_x[1], n_x).reshape(n_x, 1)

        # Covariance matrix
        self.mu = np.vstack([[self.mu_y], self.mu_R])  
        self.Sigma = np.block([
            [self.sigma_y**2, self.rho_y.T @ self.Sigma_R],
            [self.Sigma_R @ self.rho_y, self.Sigma_R]
        ])  

        # Draws for income and returns
        self.w, self.u = generate_draws(n_rand, self.N+1, rule=integration_rule)
        x = inverse_multinormal(self.u, self.mu, self.Sigma)
        self.y = np.exp(x[0,:]).reshape(1,-1)  
        self.R = x[1:,:]

    def __str__(self):
        '''String representation of the PortfolioChoiceModel'''

        summary = f"""
Model from portfolio class with attributes:
------------------------------------------------------------
Discount factor, beta                   = {self.beta}
CRRA utility parameter, gamma           = {self.gamma}
Mean log(income), mu_y                  = {self.mu_y}
Std. dev. of log(income), sigma_y       = {self.sigma_y}
Mean income, E(y)=exp(mu_y+sigma_y^2/2) = {np.exp(self.mu_y + self.sigma_y**2/2):.5f}
Std. dev. of income, std(y)             = {np.sqrt((np.exp(self.sigma_y**2)-1)*np.exp(2*self.mu_y+self.sigma_y**2)):.5f}
Number of assets, N                     = {self.N}
Mean return on assets, mu_R.T            = {np.array2string(self.mu_R.T, separator=", ")}
Covariance matrix of returns, Sigma_R   =
{np.array2string(self.Sigma_R, precision=5, separator=", ")}
Correlation vector (log income and asset return), rho_y.T =
{np.array2string(self.rho_y.T, precision=5, separator=", ")}
Covariance matrix, Sigma                =
{np.array2string(self.Sigma, precision=5, separator=", ")}
Cash-on-hand bounds, bnd_x              = {self.bnd_x}
Number of grid points for cash on hand, n_x = {self.n_x}
Number of quadrature points for income and asset returns, n_rand = {self.n_rand}
Income draws, y.shape                   = {self.y.shape}
Asset return draws, R.shape              = {self.R.shape}

Model id = {hex(id(self))}
        """
        return summary.strip()

    def utility(self, c): 
        '''CRRA utility function'''
        if self.gamma!=1:
            return (c**(1-self.gamma)-1)/(1-self.gamma)+1
        else:
            return np.log(c)+1    

    def returns(self, w):
        '''Compute portfolio returns for a given choice of weights w'''
        
        # Ensure w is a column vector (N x 1)
        w = np.array(w).reshape(self.N, 1)  # Portfolio weights (N x 1)

        # Compute portfolio return: R_t = w' * R_t+1
        R_portfolio = w.T @ self.R  # Shape: (1 x n_rand)

        return R_portfolio

    def transition(self, x, c, w):
        '''Compute next-period cash-on-hand x_{t+1} given x_t, c_t, and portfolio choice w_t'''
        
        # Ensure inputs are in correct shape
        x = np.array(x).reshape(-1, 1)  # Reshape x to (n_x,1) for vectorized computation
        c = np.array(c).reshape(-1, 1)  # Reshape c to (n_x,1)
        
        # Compute portfolio returns for given weights w (1 x n_rand)
        R_next = self.returns(w)  
        
        # Compute next-period cash-on-hand for all Monte Carlo draws (n_x x n_rand)
        x_next = (x - c) @ R_next + self.y  # Broadcasting ensures correct shapes
        return x_next  # Shape: (n_x, n_rand)

    def vf(self, choices, interpV):
        '''Compute the value function for given choices (c, w) and interpolated value function'''
        
        # Parse choices
        c = choices[0]  # Consumption choice (scalar)
        w = choices[1:]  # Portfolio allocation (vector)

        # Compute next-period cash-on-hand
        x_next = self.transition(self.x0, c, w)  # Shape: (n_x, n_rand)

        # Compute expected future value by taking expectation over next-period states
        EV = np.sum(self.w * interpV(x_next), axis=1, keepdims=True)  # Shape: (n_x, 1)

        # Compute total value function
        return self.utility(c) + self.beta * EV  # Shape: (n_x, 1)

    def bellman(self, V0):
        '''Bellman operator: Computes updated value function and optimal policy choices'''
        
        # Create interpolated value function from V0
        # interpV = interpolate.interp1d(self.x0[:, 0], V0, bounds_error=False, kind='linear', fill_value='extrapolate')
        interpV = interpolate.interp1d(self.x0[:, 0], V0, kind='cubic', bounds_error=False, fill_value='extrapolate')
        # Initialize storage for new value function and policy choices
        V1 = np.zeros_like(V0)  # New value function
        policy = np.zeros((self.n_x, 1+ self.N))  # Optimal policy function (0: consumption, 1: portfolio weights)

        # Loop over each grid point in the state space
        for i, x in enumerate(self.x0.flatten()):  # Ensure x is a scalar
            
            # Define objective function: NEGATIVE because we maximize
            obj = lambda choices: -self.vf(choices, interpV)[i]

            # Initial guess: Consume half of cash and equal portfolio weights
            if i==0:  # At the first grid point
                c_init = float(x)  # Assume consuming all cash (credit limit)
                w_init = np.ones(self.N) / self.N  # Equal allocation across assets
                choices_init = np.concatenate(([c_init], w_init))  # Correct concatenation  
            else:
                choices_init=policy[i-1, : ]*0.95 # Use previous period policy as initial guess
            # Constraints: c in [0,x] and portfolio weights sum to 1, w_i >= 0
            cons = [
                {'type': 'ineq', 'fun': lambda choices: x - choices[0]},  # c <= x
                {'type': 'ineq', 'fun': lambda choices: choices[0]-0.001},  # c >= 0
                {'type': 'eq', 'fun': lambda choices: np.sum(choices[1:]) - 1},  # sum(w) = 1
                {'type': 'ineq', 'fun': lambda choices: choices[1:]}  # w_i >= 0
            ]

            # Solve the optimization problem
            result = minimize(obj, choices_init, method='SLSQP', constraints=cons, options={'ftol': 1e-10, 'maxiter': 200})
            # result = minimize(obj, choices_init, method='trust-constr', constraints=cons)

            

            # Store the results
            V1[i] = -result.fun  # Maximum value
            policy[i, : ] = result.x    # Optimal decisions

        return V1, policy # Value function and policy functions

# Utilities for generating draws and inverse transform sampling
def generate_draws(n=3, d=2, rule="random"):
    '''Generate draws for income and asset returns
    Parameters:
        n: number of draws (scalar)
        d: number of dimensions (scalar)
        rule: quadrature rule for income and asset returns
            (str) "legendre", "random", "halton", "sobol"
    Returns:
        w: weights for quadrature (1 x N) or weight w=1/n for random draws (scalar)
        x: draws for income and asset returns (d x n)
        for muldimensional case, quadrature nodes are tensor product of 1D nodes
        with int(n**(1/d)) nodes in each dimension
    '''
    distribution = chaospy.Iid(chaospy.Uniform(0, 1), d)
    if rule == 'legendre':
        order = int(n**(1/d)) - 1
        x, w = chaospy.generate_quadrature(order, distribution, rule=rule, sparse=False)
    else:
        x = chaospy.generate_samples(n, domain=d, rule=rule)  
        w = np.full((1, x.shape[1]), 1/x.shape[1])  # Equal weight for Monte Carlo
    return w, x

def inverse_multinormal(u, mu=None, Sigma=None):
    '''Inverse transform sampling for multivariate normal
    Parameters:
        u: uniform random numbers (d x n)
        mu: mean vector (d x 1)
        Sigma: covariance matrix (d x d)
    Returns:
        x: draws from multivariate normal (d x n) with x~N(mu, Sigma)
    '''
    x = norm.ppf(u)  # inverse cdf of standard normal
    if Sigma is not None:
        L = np.linalg.cholesky(Sigma)  
        x = L @ x # 
    if mu is not None:
        mu = np.array(mu).reshape(-1,1)
        x += mu
    return x

def build_sigma(sigma = [1,.1], c=0): 
    '''build d-dimensional (dxd) covariance matrix Sigma with correlation c and std deviations sigma
    Parameters:
        sigma: list of std deviations for d random variables
        c: correlation coefficient between random variables (default=0) can only create negative correlation for d=2'''
    sigma=np.array(sigma).reshape(-1,1); 
    d=len(sigma);
    if (c<0.0) and not (d==2):
        raise RuntimeError('build_sigma can only create negaive correlation for d=2')
    corr=(1-c)*np.identity(d) + c*np.ones((d,d)); # correlation martrix 
    Sigma=sigma*corr*sigma.T # Covariance matrix  
    return Sigma

# Utilities for plotting
def v_c_plot(x, V, policy):
    '''Illustrate solution'''
    c=policy[:,0,:] # Consumption policy
    fig1, (ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
    ax1.grid(which='both', color='0.65', linestyle='-')
    ax2.grid(which='both', color='0.65', linestyle='-')
    ax1.set_title('Value function')
    ax2.set_title('Consumption policy function')
    ax1.set_xlabel('Cash on hand, x')
    ax2.set_xlabel('Cash on hand, x')
    ax1.set_ylabel('Value function')
    ax2.set_ylabel('Consumption function')
    if len(V.shape)==1: 
        V=V[:,np.newaxis]
        c=c[:,np.newaxis]
    for i in range(V.shape[1]):
        ax1.plot(x[1:],V[1:,i],color='k',alpha=0.25)
        ax2.plot(x[1:],c[1:,i],color='k',alpha=0.25)
    # add solutions
    ax1.plot(x[1:],V[1:,0],color='r',linewidth=2.5)
    ax2.plot(x[1:],c[1:,0],color='r',linewidth=2.5)
    plt.show()

def w_plot(x, V, policy):
    '''Illustrate solution for portfolio shares'''
    w = policy[:, 1:, :]  # Portfolio shares (excluding consumption)
    
    num_assets = w.shape[1]  # Number of assets in portfolio
    fig, ax = plt.subplots(1, num_assets, figsize=(4 * num_assets, 4))  # One subplot per asset
    
    if num_assets == 1:  # Ensure ax is iterable even for a single asset
        ax = [ax]

    for j in range(num_assets):
        ax[j].grid(which='both', color='0.65', linestyle='-')
        ax[j].set_title(f'Portfolio share $w_{j+1}(x)$')
        ax[j].set_xlabel('Cash on hand, x')
        ax[j].set_ylabel(f'Portfolio share $w_{j+1}$')

        if len(V.shape) == 1:  # Ensure proper shape handling
            V = V[:, np.newaxis]
            w = w[:, :, np.newaxis]

        for i in range(V.shape[1]):  # Iterate over solution iterations
            ax[j].plot(x[1:], w[1:, j, i], color='k', alpha=0.25)  # Plot historical solutions
        
        # Highlight the first iteration in red
        ax[j].plot(x[1:], w[1:, j, 0], color='r', linewidth=2.5)

    plt.show()


