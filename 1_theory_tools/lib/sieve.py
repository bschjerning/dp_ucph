import numpy as np
import matplotlib.pyplot as plt

# for printing reasonable nice output for matrics
def disp(arr,fmt="{:8.4f}"):
    for row in arr:
        for item in row:
            print(fmt.format(item), end = " ")
        print("")

def expand(k, x):
    x=np.atleast_1d(x)
    if x.size==1:
        x= np.full(k, x)
    return x 

# plotting tool for f(x), data (x0,y0), and our approximation fhat(x) on the interval [a,b]
def plot1d(f=None, x0=None, fx0=None, fhat=None, a=None, b=None, color='b',label=''):
    '''helper function to make plots'''
    plt.figure(num=1, figsize=(10,8))
    if a is None: a=np.min(x0); 
    if b is None: b=np.max(x0);
    x = np.linspace(a,b,1000) # for plotting func of interst on [a,b]
    if fx0 is not None : plt.scatter(x0,fx0,color='r') # interpolation data
    if f is not None: 
        plt.plot(x,f(x),color='grey', label='f(x)') # true function
    if fhat is not None:
        plt.plot(x,fhat(x),color=color,label=label)
    if (f is not None)  or (fhat is not None):
        plt.legend()

def plot2d(X,y, n, fig=None, i=1, label='f(x1,x2)'): 
    if fig is None: fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1, 3, i, projection='3d')
    x1=X[:,0].reshape(n[0], n[1])
    x2=X[:,1].reshape(n[0], n[1])
    y=y.reshape(n[0], n[1])
    ax.plot_surface(x1, x2, y, cmap=plt.cm.Spectral, linewidth=0, antialiased=False)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel(' ')
    ax.set_title(label)
    ax.view_init(-140, 35)
    return fig

            
class sieve():
    '''Class to approximate functions by linear sieves'''
    def __init__(self, n=6, deg=None, a=0, b=1, btype='chebyshev', gridtype='c', nknots=None,
        transform =None, inv_transform =None):
        '''Initializer'''
        self.transform=transform
        self.inv_transform=inv_transform
        self.n=np.atleast_1d(n)
        if deg is None: deg=self.n-1       # interpolation is default
        if nknots is None: nknots=self.n + 1- np.maximum(deg,1) # interpolation is default
        self.k=len(self.n);
        
        for arg in ('deg', 'a', 'b', 'btype', 'gridtype', 'nknots'): 
            setattr(self, arg, expand(self.k, eval(arg)))

        # Generate gridpoints
        self.x,self.z =sieve.grid(self.n, self.a, self.b, self.gridtype)

        # uniform internal knots
        self.knotsx,self.knotsz =sieve.grid(self.nknots, self.a, self.b, expand(self.k, 'u'))
        B = sieve.basis(self.z, self.deg, self.btype, self.knotsz)
        self.B=B
        self.pB=np.linalg.inv(B.T@B)@B.T        
    
    def grid(n=10, a=0, b=1, gridtype='u'): # grid
        ''' multidimensional catesian grid:       
        parameters:
            a:         k-dimensional vector or scalar with mininimum values in grid (if scalar, same min is assumed for all s)
            b:         k-dimensional vector or scalar with max values in grid (if scalar, same max is assumed for all s)
            n:         k-dimensional vector holding number of grid points in each dimension of X
            gridtype:  k- or 1-dimensional vector with gridtype (if one dimensional same gridtype is assumed for all s)
                       {'u'} for uniform grids, {'c'} for chebyshev nodes
        returns:
           x: grid defined on X in {[a[0],b[0]] x [a[1],b[1]] x ... x [a[k-1],b[k-1]]} with n[j] nodes in each dimension of X
           z: grid defined on Z in {[-1,1]^s} with n(j) nodes in each dimension of Z

          EXAMPLE:     [x,z]=sieve.grid(n=[6 3 2],a=1,b=2,['u','c','c']) produces a grid where the first variable is spanned by on the interval [a,b]=[1,2]
                       uniform grid with 6 elements, and the the second and the third is spanned by chebyshev nodes with 3 and 2 nodes respectivily. 
                       x is a (6*3*2) x 3 matrix.
        '''
        
        n=np.atleast_1d(n)
        a=expand(len(n), a)
        b=expand(len(n), b)
        gridtype=expand(len(n), gridtype)
                
        z=[];x=[];
        for j, tp in enumerate(gridtype):
            if tp=='u':   
                z0 = np.linspace(-1,1,n[j])
            elif tp=='c': 
                z0 =-np.cos((np.linspace(1,n[j],n[j])-0.5)*np.pi/n[j])   # nodes on [-1,1]
            elif tp=='ce': # Expanded chebyshev 
                z0 =1/np.cos(np.pi/(2*n[j]))*np.cos(((2*np.linspace(1,n[j],n[j])-1)/2)*np.pi/n[j])   # nodes on [-1,1]
            elif tp=='rand': 
                z0 = 2*np.sort(np.random.random(n[j]))-1                 # nodes on [-1,1]
            else: raise RuntimeErrort('gridtype not implmeneted')
            x0=(z0+1)*((b[j]-a[j])/2)+a[j];                              # nodes on [a,b]
            x.append(x0);
            z.append(z0);
            
        z=sieve.cartesian(z)
        x=sieve.cartesian(x)    

        return x, z
    
    def fit(self, y, x=None):
        y=y.reshape(-1,1)
        if self.transform: 
            y=self.transform(y)
        if x is not None:
            # Generate gridpoints
            z=2*(x-self.a.T)/(self.b.T-self.a.T)-1
            B = sieve.basis(z, self.deg, self.btype, self.knotsz)
            return np.linalg.inv(B.T@B)@B.T@y    
        else:
            return self.pB@y

    def eval(self, x, c):
        b=sieve.basis(self.x2z(x), self.deg,  self.btype, self.knotsz)
        yhat=b@c
        if self.inv_transform: 
            yhat=self.inv_transform(yhat)
        return yhat

    def x2z(self, x):
        return 2*(x-self.a.T)/(self.b.T-self.a.T)-1
        
    def cartesian(x):
        '''cartesian product: combination of all elements in the list of vectors x=[x[0],x[1],...,x[k-1]]
         Parameters:
           x=[x[0],x[1],...,x[k-1]] where x[j] is a (n[j],1) column vector
         returns:
           X:  (prod(n),k) matrix with cartesian with combination of all elements in the list of vectors'''
        d=len(x)
        return np.array(np.meshgrid(*x)).T.reshape(-1,d)
    
    def tensor(b): 
        '''k-fold tensor product list of matrices b=[b[0],b[1],...,b[k-1]]
         parameters:
           b=[b[0],b[1],...,b[k-1]] where b[j] is a (m x n[j]) matrix of basis functions for dimension j
         returns:
           T:   (m x (prod(n)) matrix holding k-fold tensor product of list of matrices b=[b[0],b[1],...,b[k-1]]'''
        T=b[0]   # Initialize Tensor product basis with first element in b 
        for j in range(len(b)-1): # loop over remainng dimensions in b
            T=(T[:,:,None]*b[j+1][:,None,:]).reshape(T.shape[0],-1)  # use boradcasting
        return T
    
    def basis(z, deg, btype, knots=None):
        B=[]
        k=z.shape[1]
        for j in range(k): 
            if (knots is None):   
                Bj=sieve.basis_j(z[:,j], deg[j], btype[j])
            else:
                Bj=sieve.basis_j(z[:,j], deg[j], btype[j], knots[:,j])
            B.append(Bj)
        return sieve.tensor(B) 
            
    def basis_j(z, deg, btype='chebyshev',knots=None):
        z=z.reshape(-1,1) # x m,ust be column vector
        n=z.size
        if btype=='algpol':      # Simple powers of x: [x^0, x^1, x^2, ... x^deg]
            b=z**np.arange(deg+1) 
        elif btype=='chebyshev': # Chebyshev polynomials of degree 0-deg: [T_0(x), T_1(x), T_2(x), ... T_deg(x)]
            b=np.cos(np.arange(deg+1)*np.arccos(z)) 
        elif btype=='b-spline': 
            b=sieve.splinecol(knots,deg+1,z);
        else: 
            raise RuntimeError('basis not implmeneted')
        return b

    def splinecol(knots,k,x):
        '''k order b-spline, knots: vector of INTERBAL knots, x where to evaluate Bspline ''' 

        knots=knots.reshape(-1,1) # x m,ust be column vector
        x=x.reshape(-1,1) # x m,ust be column vector
        x=x[:,0]
    
        n=x.shape[0]
        nknots=knots.size
        N=np.zeros((n,nknots-k));

        x_u,x_ia,x_ic= np.unique(x,return_index=True,return_inverse=True, axis=0)
        u, knots_ia, knots_ic= np.unique(knots,return_index=True,return_inverse=True, axis=0)
        u=u.reshape(-1,1); 
        nu=u.size
        n_u=x_u.shape[0]
        u[-1]=u[-1]+1e-8;

        if k>1:
            u=np.concatenate( (np.ones((k-1,1))*u[0], u,  np.ones((k-1,1))*u[-1]), axis=0)        

        nu=u.size
        Nu=np.zeros((n_u,nu-1))
        for i in range(k-1,nu-1,1): 
            Nu[:,i]= (x_u>=u[i])*(x_u<u[i+1])
            
        for p in range(k-1): 
            for i in range(nu-p-2):  
                if u[i+p+2]==u[i]:
                    Nu[:,i]=0
                elif (u[i]<u[i+p+1]) and (u[i+1]==u[i+p+2]):                               
                    s1= (x_u-u[i])/(u[i+p+1]-u[i])*Nu[:,i]
                    Nu[:,i]= s1
                elif u[i]==u[i+p+1] and u[i+1]<u[i+p+2]:
                    s2= (u[i+p+2]-x_u)/(u[i+p+2]-u[i+1])*Nu[:,i+1]
                    Nu[:,i]= s2
                else:
                    s1= (x_u-u[i])/(u[i+p+1]-u[i])*Nu[:,i]
                    s2= (u[i+p+2]-x_u)/(u[i+p+2]-u[i+1])*Nu[:,i+1]
                    Nu[:,i]=s1+s2
        if k>1:
            N=Nu[x_ic,:-k+1]   
        else: 
            N=Nu[x_ic,:]        
        return N

    def plot1d(self, f=None, color='b',label=None): 
        if label is None: label=self.btype[0] + ' (n=' + str(self.n[0]) + ', deg=' + str(self.deg[0]) + ')'
        '''helper function to make plots'''
        fig, ax = plt.subplots(1,2, figsize=(15,5))
        x,z = sieve.grid(1000, self.a, self.b, self.gridtype) 
        fx=f(x)
        x0=self.x
        fx0=f(x0)
        
        α = self.fit(fx0)                  # sieve coeficients
        
        ax[0].plot(x,fx, color='grey', label='f(x)') # true function
        ax[0].scatter(x0,fx0,color='r', label='interpolation data') # interpolation data 
        ax[0].plot(x,self.eval(x, α),color=color,label=label)
        ax[0].legend()
        ax[1].scatter(x0,self.eval(x0, α)- fx0,color='r', label='interpolation data') # interpolation data 
        ax[1].plot(x,self.eval(x, α)-fx,color=color,label=label)
        ax[1].set_title('Max abs. approximation error %-10.2g' % np.max(np.abs(self.eval(x, α)-fx)))
        ax[1].legend()
    
