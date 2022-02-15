import numpy as np
import lib.integrate as intpy
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm

class portfolio():
    '''Portfolio choice model class'''
    def __init__(self, 
        gamma=-2, 
        W= 1,
        R=np.array([1,2]), 
        sigma2_Z=np.array([0,1])**2,
        corr=0,
        m=10,  
        integration_method="quadrature",
        verbose=1
        ): 
        '''Initializator for the portfolio class'''

        if np.isscalar(sigma2_Z): "sigma2_Z must be list or array"
        if np.isscalar(R): "R must be list or array"
        assert sigma2_Z.shape==R.shape, "Shape of R and sigma2_Z does not add up"
        self.n = len(R) # number of assets
        sigma2_Z.shape=(self.n,1);
        R.shape=(self.n,1);


        if np.isscalar(m) & (integration_method=="quadrature"): 
            m=np.ones(len(R))*m
            m=m.astype(int)

        self.gamma = gamma  # CRRA utility parameter
        self.W = W  # CRRA utility parameter
        self.R = R  # Mean return on assets
        self.sigma2_Z = sigma2_Z  # varaince of return on assets
        self.mu=np.log(R**2/((R**2 + sigma2_Z)**0.5))
        std=np.log(1+(sigma2_Z/R**2))**0.5
        if np.isscalar(corr): 
            self.corr=np.identity(self.n)
            self.sigma=np.diag(std[:,0])
        else:
            self.corr=corr
            sigma2=std*self.corr*std.T                  # covariance martrix
            self.sigma=np.linalg.cholesky(sigma2)       # Lower triangular cholesky matrix


        self.integration_method=integration_method
        if integration_method=="quadrature":
            self.x,self.w=intpy.quad_xw(m,0, 1)
        elif integration_method=="monecarlo":
            rng = np.random.default_rng()
            self.x =np.random.rand(n,m)
            self.w = 1/m

        self.m = m      # number of quadrature nodes
        
        if verbose: 
            print(self)

    @property
    def m(self):
        '''Dimension getter'''
        return self.__m # internal dimension variable

    @m.setter
    def m(self,m):
        '''Dimension setter, updating integration points'''
        self.__m = m
         
        if self.integration_method=="quadrature":
            self.x,self.w=intpy.quad_xw(m,0, 1)
        elif self.integration_method=="montecarlo":
            rng = np.random.default_rng()
            self.x =np.random.rand(self.n,m)
            self.w = 1/m

    def __str__(self):
        '''String representation of the sgame model object'''
        # id() is unique identifier for the variable (reference), convert to hex
        print('Model from portfolio class with attributes:')
        print('-'*60)
        print('CRRA utility parameter, gamma           = ', self.gamma, )
        print('Initial wealht, W                       = ', self.W)
        print('Mean return on assets , R.T               = ', self.R.T)
        print('Std. dev. on return on assets, sigma_Z.T  = ', self.sigma2_Z.T**0.5)
        print('Number of assets, n                     = ', self.n)        
        print('Number of quad nodes for each assset, m = ', self.m)
        print('Correlation matrix, (log asset return), Cor:\n',self.corr.round(4))
        print('Cholesky matrix, (log asset return), Sigma:\n',self.sigma.round(4))
          
        return 'Model id = %s\n' % hex(id(self))
    
    def __repr__(self):
        '''Print for sgame model object'''
        return self.__str__()

    def Z(self, x):
        # distribution of asset returns
        return np.exp(self.mu+self.sigma @ norm.ppf(x))   

    def E_Zi(self): 
        '''Expected return of asset iZ'''
        return np.sum(self.w*self.Z(self.x), axis=1)

    def E_u(self, omega, use_derivatives=0):
        omega=np.append(self.W-np.sum(omega), omega)
        omega.shape=(len(omega),1)

        Z=self.Z(self.x);

        c=omega.T @ Z
        u=np.sum(self.w*self.u(self.gamma, c), axis=1)

        if use_derivatives==0:
            return u
        else:
            g=np.zeros(len(omega)-1)
            H=np.zeros((len(omega)-1, len(omega)-1))
            wdu=self.w*self.du(self.gamma, c)
            wddu=self.w*self.ddu(self.gamma, c)
            for i in range(len(omega)-1):
                g[i]= wdu @  (Z[[i+1],:]-Z[[0],:]).T  
                for j in range(i+1):
                    H[i, j]= wddu @ ((Z[[i+1],:]-Z[[0],:])*(Z[[j+1],:]-Z[[0],:])).T
                    H[j, i]= H[i, j]

            return u, g, H

    def u(self, gamma, c): 
        '''CRRA utility function'''
        if gamma!=1:
            return (c**(1-gamma)-1)/(1-gamma)+1
        else:
            return np.log(c)+1
    
    def du(self, gamma, c): 
        '''CRRA utility function'''
        return (c**(-gamma))

    def ddu(self, gamma, c):
        '''CRRA utility function'''
        return (c**(-gamma-1))*(-gamma)

    def plt_pdf_Z(self, n=1000, figname=None):
        x = np.linspace(0, 10, n)
        fig, ax = plt.subplots(1, 1,figsize=(8,5))

        for i in range(self.n):
            mu=self.mu[i]
            sigma=self.sigma[i, i]
            ax.plot(x, lognorm.pdf(x, s=sigma, scale=np.exp(mu)), label=r'$\mu=$%g, $\sigma=$%g,  $R=E(Z)=$%g, $\sigma^2_{Z}=Var(Z)=$%g' % (mu, sigma, self.R[i], self.sigma2_Z[i]))


        ax.set_title('Probability density function for log normal, $ln(Z) \sim N(\mu, \sigma^2)$')
        ax.legend()
        ax.set_xlabel('$Z, $')
        ax.set_ylabel('$pdf(Z)$')
        if figname: 
            plt.savefig(fname=figname, dpi=150)

    def plt_E_Z(self,mvec=range(2, 20), figname=None):         
        model=self
        E_Z=np.zeros((len(mvec), len(model.R)))

        for i_m in range(len(mvec)):
            model.m=[1, mvec[i_m]]
            E_Z[i_m,:]=model.E_Zi().T

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        ax1.set_title('Expected value of assets, $E(Z)$')
        ax1.set_xlabel('Number of quadrture points, $m$')
        ax1.set_ylabel('$E(Z)$')
        for i_Z in range(len(model.R)):
            ax1.plot(mvec, E_Z[:,i_Z], label=r'$R_%d=$%g, $\sigma^2_{Z_%d}=$%g' % (i_Z, model.R[i_Z], i_Z, model.sigma2_Z[i_Z]))
        ax1.legend()

        if figname: 
            plt.savefig(fname=figname, dpi=150)

        for i_Z in range(len(model.R)):
            ax2.plot(mvec, E_Z[:,i_Z]-model.R[i_Z], label=r'$R_%d=$%g, $\sigma^2_{Z_%d}=$%g' % (i_Z, model.R[i_Z], i_Z, model.sigma2_Z[i_Z]))

        ax2.set_title('Approximation error for expected value of assets, $E(Z)-R$')
        ax2.legend()
        ax2.set_xlabel('Number of quadrture points, $m$')
        ax2.set_ylabel('$E(Z)-R$')
        plt.savefig(fname=figname, dpi=150)

    def plt_E_U(self,xvec=range(2, 20), xvar="nodes", gammas=[1,1.5,2], omega=0.5, mfine=10, figname=None):         
        model=self

        fig, ax1 = plt.subplots(1, 1,figsize=(6,4))
        ax1.set_title('Expected utility of asset portfolio, $E_m(U(\omega_1))$')
        if xvar=="nodes": 
            ax1.set_xlabel('Number of quadrature points, $m$')
        elif xvar=="omega": 
            ax1.set_xlabel('Share in asset_1, $omega_1$')
                
        ax1.set_ylabel('$E_m(U(\omega_1))$')
        for gamma in gammas:
            model.gamma=gamma
            E_U=np.zeros((len(xvec), 1))
            for i_x in range(len(xvec)):
                if xvar=="nodes": 
                    model.m=[xvec[i_x], xvec[i_x]]
                elif xvar=="omega": 
                    omega=[xvec[i_x]]

               
                E_U[i_x]=model.E_u(omega)

            ax1.plot(xvec, E_U[:],  label=r'$\gamma=$%g' % gamma);
        ax1.legend()
        if figname: 
            plt.savefig(fname=figname, dpi=150)

    def plt_u(self, gammas=None, c=np.linspace(0.2, 10, 100),  figname='u.png'):

        if gammas==None:
            gammas=[self.gamma]

        # Plot best response functions
        
        fig1, ax = plt.subplots(1,1,figsize=(8,8))
        ax.set_title('Utility function, $u(c)=(c^{1-\gamma}-1)/(1-\gamma)$')
        for gamma in gammas:
            ax.plot(c, self.u(gamma,c),  label=r'$\gamma=$%g' % gamma);

        ax.legend()
        ax.set_xlabel('$Consumption, c$')
        ax.set_ylabel('$Utility, u(c)$')
        # ax.set_xlim(0.0, 1)
        # ax.set_ylim(0.0, 1)
        plt.savefig(fname=figname, dpi=150)






        
