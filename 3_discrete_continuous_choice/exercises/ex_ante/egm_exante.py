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

        # Futute m and c
        

        # Future marginal utility

        # Currect C and m
        #sol.c[t,i_a+1]=
        #sol.m[t,i_a+1]=

    return sol

def EGM_vec (sol,t,par):

    

    return sol
