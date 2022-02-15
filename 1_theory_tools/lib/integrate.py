import numpy as np

'''Integrate module'''

def quad_xw(n=10, a=-1, b=1):
    '''
    quad_wx: Procedure to compute quadrature weights and nodes to integrate function f([x1,...,xd])
             on interval a[], b[i] with n[i] nodes in each dimension i=1,...,d=len(n)    
             
    parameters
      n:     1d array with shape (d) where each element holds the number of nodes in each dimension d
      a, b:  lower, upper integration limits (scalar or array with same shape as n)

    outputs
      x  :  1d or 2d array with shape (m,d): nodes for x1, x2, ..., xd   
      w  :  1d array with shape (m,1)  '''
    
    if np.isscalar(n):
        n=np.ones((1,1))

    d=len(n)

    if np.isscalar(a):
        a=np.ones(d)*a
    if np.isscalar(b):
        b=np.ones(d)*b
    x=[];
    w=[];
    for i in range(d):
        xi, wi = np.polynomial.legendre.leggauss(n[i])
        w.append((b[i]-a[i])/2*wi)
        x.append((xi+1)*(b[i]-a[i])/2+a[i])

    x =np.array(np.meshgrid(*x)).T.reshape(-1,d)
    w =np.array(np.meshgrid(*w)).T.reshape(-1,d)
    w=np.prod(w, axis=1,  keepdims=True)

    return x.T,w.T
        
