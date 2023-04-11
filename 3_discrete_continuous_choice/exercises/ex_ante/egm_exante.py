# Import package and module
import numpy as np
import utility as util
import tools


def EGM(sol,t,par):
    sol = EGM_loop(sol,t,par) 
    #sol = EGM_vec(sol,t,par) 
    return sol

def EGM_loop (sol,t,par):
    for i_a,a in enumerate(par.grid_a[t,:]):

        # Fill in:
        # Hint: Same procedure as in 02_EGM.ipynb
        
        # Future m and c
        if t+1<= par.Tr: # No pension in the next period
            fac = par.G*par.L[t]*par.psi_vec # Trick to ease notation and calculations

        else:
            fac = par.G*par.L[t]

        # Future expected marginal utility

        # Current C and m
        #sol.c[t,i_a+1]=
        #sol.m[t,i_a+1]=

    return sol

def EGM_vec (sol,t,par):

    #Fill in:
    

    return sol
