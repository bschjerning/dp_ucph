
# %%
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from scipy import interpolate

class DeatonEGM:
    def __init__(self, beta=0.95, R=1.05, μ=1, σ=1, η=1, Mbar=10, ngrid=5, n_y=10):
        """Initialize model parameters and grids"""
        # Model parameters
        self.beta = beta  # Discount factor
        self.R = R  # Return on savings
        self.μ = μ  # Mean of log income
        self.σ = σ  # Std dev of log income
        self.η = η  # CRRA parameter
        
        # Technical parameters
        self.Mbar = Mbar  # Max cash on hand
        self.ngrid = ngrid  # Number of grid points
        self.n_y = n_y  # Number of quadrature points
        
        # Utility functions
        self.util = lambda c: np.log(c) if η == 1 else (c**(1-η) - 1) / (1-η)
        self.mutil = lambda c: c**(-η)  # Marginal utility
        self.imutil = lambda mu: mu**(-1/η)  # Inverse marginal utility
        
        # Grid for assets
        self.A = np.linspace(0, Mbar, ngrid)
        
        # Quadrature grids for y (adjusted weights and nodes)
        q, w = np.polynomial.legendre.leggauss(n_y)  # Legendre quadrature on [-1,1]
        Ginv = lambda z: np.exp(scipy.stats.norm.ppf(z, loc=μ, scale=σ))  # Log-normal inverse CDF
        self.weights = w / 2  # Adjust weights for [0,1] interval
        self.y = Ginv((q + 1) / 2)  # Transform nodes to log-normal space
        
        # Initial conditions
        self.Aex = np.full(ngrid + 1, np.nan)
        self.Aex[1:] = self.A
        
    def egm_iter(self, M0, c0):
        """Perform one iteration of the EGM algorithm"""
        policy = interpolate.interp1d(M0, c0, kind='slinear', fill_value="extrapolate")  # Interpolation function

        M1 = np.full(self.ngrid + 1, np.nan)
        c1 = np.full(self.ngrid + 1, np.nan)

        M1[0] = c1[0] = 0  # Ensure first point at origin

        for j, aj in enumerate(self.A):
            Mpr = np.maximum(self.R * aj + self.y, 1e-10)  # Next period wealth
            cnext = policy(Mpr)  # Next period consumption
            Emu1 = self.beta * self.R * np.sum(self.weights * self.mutil(cnext))  # Expected marginal utility
            c = self.imutil(Emu1)  # Inverse Euler equation
            M = aj + c  # Endogenous wealth
            
            M1[j + 1] = M
            c1[j + 1] = c

        return M1, c1, self.plot_iter(M1, c1)
    
    def plot_iter(self, M1, c1):
        """Plot EGM iteration results"""
        fig, ax = plt.subplots()
        ax.plot(M1, c1, marker='o', linestyle='-', label="Policy function")
        ax.set_xlabel("Cash on Hand (M)")
        ax.set_ylabel("Consumption (c)")
        ax.set_title("Endogenous Grid Method (EGM) Iteration")
        ax.legend()
        return fig  # Return figure object for further use if needed

# Example usage
model = DeatonEGM()
M0 = np.array([0, model.Mbar])  # Initial wealth grid
c0 = np.array([0, model.Mbar])  # Initial consumption policy
M1, c1, plot = model.egm_iter(M0, c0)  # Run one EGM iteration
for i in range(5):
    M1, c1, plot = model.egm_iter(M1, c1)  # Run more iterations
plot.show()
# %%
