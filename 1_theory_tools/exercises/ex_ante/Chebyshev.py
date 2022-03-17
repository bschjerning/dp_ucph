import numpy as np


def Chebyshev(fhandle,points,m,n):
    
    # This is the Chebyshev Interpolation (Regression algorithm)      
    #  in approximation of a scalar function, f(x):R->R                
    #    The approach follow Judd (1998, Allgortihm 6.2, p. 223)         
#############################################################################
# INPUT ARGUMENTS:
#             fhandle:               The funtion, that should be approximated
#             interval:              The interval for the approximation of f(x).
#             m:                     number of nodes used to construct the approximation. NOTE: m>=n+1
#             n:                     Degree of approximation-polynomial
# 
# OUTPUT ARGUMENTS:
#             f_approx:              The vector of approximated function values
#             f_actual:              The vector of actual function values
#             points:                The vector of points, for which the function is approximated
##################################################################################


    assert (m>=n+1), 'The specified parameters are not acceptable. Make sure m>n'

    a = points[0]
    b = points[-1]
    number = points.size
    f_approx = np.nan + np.zeros((number))  # Initial vector to store the approximated function values
    f_actual = np.nan + np.zeros((number))  # Initial vector to store the actual function values

    for x in range(number):                   # Loop over the x values
        ai = np.nan +np.zeros((n+1))         # Initial vector to store the Chebyshev coefficients
        f_hat = 0                             # Initial value of the approximating function
        for i in range(n+1):                  # Loop over the degree of the approximation polynomial. 
            nom = 0                           # Initial value for step 4
            denom = 0                         # Initial value for step 4
            for k in range(m):                # Loop over the approximation notes
                
                # Step1: Compute the m Chebyshev interpolation notes in [-1,1]    
                zk = -np.cos(((2*(k+1)-1)/(2*m))*np.pi)

                # Step 2: Adjust the nodes to the [a,b] interval
                xk = (zk+1)*((b-a)/2)+a

                # Step 3: Evaluate f at the approximation notes. Loaded from f.m
                yk = fhandle(xk);  

                # Step 4: Compute Chebyshev coefficients. Tn=cos(i*acos(zk)) is loaded from Tn.m
                nom += yk*Tn(zk,i)
                denom += Tn(zk,i)**2
                if k==m-1:
                    ai[i] = nom/denom
            
            f_hat = f_hat+ai[i]*Tn(2*(points[x]-a)/(b-a)-1,i)  # The Chebyshev approximation of f(x)
            f_temp = fhandle(points[x])                       # Actual function value, f(x)

        f_approx[x] = f_hat
        f_actual[x] = f_temp

    return f_approx, f_actual, points

def Tn(x,n):
    return np.cos(n*np.arccos(x))


