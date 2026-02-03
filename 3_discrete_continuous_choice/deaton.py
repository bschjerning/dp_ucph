# %% 
import numpy as np
from scipy import interpolate # Interpolation routines
import matplotlib.pyplot as plt
from dpsolver import dpsolver
import scipy.stats
from scipy import interpolate # Interpolation routines

class deaton():
    '''Class to implement deaton's model with log-normally distrubuted income shocks'''
    def __init__(self,β=.9, R=1, μ=0, σ=1, η=1, xbar=[0, 10], n_x=50, n_c=100, n_y=10):
        '''Initializer'''
        # structural parameters
        self.β  = β       # Discount factor
        self.R  = R      # Returns on savings
        self.μ  = μ       # Location parameter for income shock, y (if y is log normal, ln(y) ~ N(μ, σ^2))  
        self.σ  = σ       # Scale parameter for income shock, y
        self.η  = η       # CRRA utility parameter, (η=0: linear, η=1: log, η>1: more risk averse than log
        
        # spapces
        self.xbar = xbar  # Upper and lower bound on cash on hand
        self.n_x = n_x    # Number of grid points for cash on hand
        self.n_c = n_c    # Number of grid points for choice grid
        self.n_y = n_y    # Number of quadrature points for income

        # quadrture grids for y (adjusted weights and nodes)
        q, w = np.polynomial.legendre.leggauss(n_y) # legendre quadrture nodes and weights on [-1,1]
        Ginv = lambda z:  np.exp(scipy.stats.norm.ppf(z, loc=self.μ, scale =self.σ))  # inverse cdf of log-normal
        self.weights=w/2; # change of varibale to adjust to [0,1] interval
        self.y = Ginv((q+1)/2) # use change of varibale to adjust nodes to [0,1] interval and use Ginv to obntain y
        
        # grids for x and c (adjusted to satisfy constraints)
        self.xbar[0] = np.maximum(np.finfo(float).eps, self.xbar[0]);  # truncate lower bound at smallest positive float number
        self.x = np.linspace(self.xbar[0],self.xbar[1],n_x).reshape((n_x,1)) # grid for state space (n_w x1 array)
        self.c = np.empty((n_x,n_c)) # initilize grid for choices for each staty point 
        for i in range(n_x): 
            # make grid between that satisfies the constraint, c\in[0,x[i]] 
            self.c[i,:] = np.linspace(self.xbar[0],self.x[i],n_c).reshape((1,n_c)) 

    def utility(self,c): # utility function
        '''Utility function, crra'''
        if self.η==1:
            return np.log(c)
        elif self.η>=0:
            return (c**(1-self.η) -1)/(1-self.η)

    def bellman(self,V0):
        '''Bellman operator, V0 is one-dim vector of values on state grid'''
        interp = interpolate.interp1d(self.x[:,0],V0,  bounds_error=False,fill_value='extrapolate')
        EV=0 
        for i, y_i in enumerate(self.y): # compte expectation wrt to shocks to future state
            x1=self.R*(self.x-self.c) + y_i # next period x, conditional of w and c, shock, nW x nC
            EV+=self.weights[i]*interp(x1);
        matV1 = self.utility(self.c) + self.β * EV        
        i_max= np.argmax(matV1,axis=1); # (column) idex of optimal choices
        V1 = matV1[np.arange(self.n_x),i_max]  
        c1 = self.c[np.arange(self.n_x),i_max] 
        return V1, c1

    def v_c_plot(self, x, V, c):
        '''Illustrate solution'''
        fig1, (ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
        ax1.grid(which='both', color='0.65', linestyle='-')
        ax2.grid(which='both', color='0.65', linestyle='-')
        ax1.set_title('Value function')
        ax2.set_title('Policy function')
        ax1.set_xlabel('Cash on hand, x')
        ax2.set_xlabel('Cash on hand, x')
        ax1.set_ylabel('Value function')
        ax2.set_ylabel('Policy function')
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

m1=deaton(β=.9, R=1.0, μ=1, σ=.1, η=3, xbar=[0, 10], n_x=100, n_c=100, n_y=10); 
m2=deaton(β=.9, R=1.0, μ=1, σ=.1, η=1, xbar=[0, 10], n_x=100, n_c=100, n_y=10); 

V, c = dpsolver.vfi(m1, maxiter=1000, callback=dpsolver.iterinfo); m1.v_c_plot(m1.x, V, c);
V, c = dpsolver.vfi_T(m1); m1.v_c_plot(m1.x, V, c);
V, c = dpsolver.vfi_T(m2); m2.v_c_plot(m2.x, V, c);

# %%
