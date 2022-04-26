
def marg_util(c,par):
    return c**(-par.rho)

def inv_marg_util(u,par):
    return u**(-1/par.rho)
