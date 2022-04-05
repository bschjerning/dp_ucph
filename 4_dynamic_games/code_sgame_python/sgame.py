import numpy as np
import matplotlib.pyplot as plt

class sgame():
    '''Simple static entry game model class'''

    def __init__(self, 
        alpha=5, # default parameter values
        beta=-11, 
        x_a=0.52, 
        x_b=0.22, 
        esr = 0,
        N=10000,
        max_sa=100,
        verbose=1
        ): 
        '''Initializator for the sgame class'''
        self.alpha = alpha  # parameter for monopoly profits 
        self.beta = beta    # parameter for duopoly profits 
        self.x_a = x_a      # size of firm a
        self.x_b = x_b      # size of firm b
        self.esr = esr      # index for equilibrium selection (0,1,2)
        self.N = N          # number of crossectional observations
        self.max_sa = max_sa  # maximum number of iterations

        if verbose: 
            print(self)

    def __str__(self):
        '''String representation of the sgame model object'''
        # id() is unique identifier for the variable (reference), convert to hex
        print('Model from sgame class with attributes:')
        print('   alpha    = %10.2f' % self.alpha, '(Monopoly profitss)')
        print('   beta     = %10.2f' % self.beta, '(Duopoly profitss)')
        print('   x_a      = %10.2f' % self.x_a, '(type, firm_a)')
        print('   x_b      = %10.2f' % self.x_b, '(type, firm b)')
        # print('   N        = %10i' % self.N, '(Number of markets to simulate)')
        return '   Model id = %s\n' % hex(id(self))
    
    def __repr__(self):
        '''Print for sgame model object'''
        return self.__str__()

    def br_a(self, p_b): 
        '''best response function, firm a'''
        p_a=1/(1+np.exp(-self.x_a*self.alpha+ p_b*self.x_a*(self.alpha-self.beta)))
        return p_a

    def br_b(self, p_a): 
        '''best response function, firm b'''
        p_b=1.0/(1+np.exp(-self.x_b*self.alpha+ p_a*self.x_b*(self.alpha-self.beta)))
        return p_b 

    def br2_a(self, p_a): 
        ''' second order best response function, firm a '''
        p_b=self.br_b(p_a)
        p_a=self.br_a(p_b) 
        return p_a

    def FindEqb(self, fx0):
        p0=self.FindStableEqb(0, fx0)
        p1=self.FindStableEqb(1, fx0)
        if abs(p0-p1)>1e-6: # more than one stable equilibrium found.
            # solve for Unstible Equilibrium using bisection algorithm
            pu=self.FindUnstableEqb(p0,p1, fx0)
            p=np.array([p0,pu,p1])
        else: # unique equilibrium
            p=np.array([p0])   
        return p

    def FindStableEqb(self, p0, fx):
        ''' Procedure to find stable equilibrium, by successive approximations  '''
        ''' inputs                                                              '''
        '''   p0: staring points                                                '''
        '''   fx: second order best response                                    '''
        ''' output                                                              '''
        '''   p: equilibrium                                                    '''
        '''   NOTE in case of multiple fixed points, p depends which p0 initializes the algorithm. '''

        # Solve for fixed point using successive approximations on the second order best response function
        for iter in range(self.max_sa):
            p=fx(p0)
            if abs(p-p0)<1e-10:
                tolerance=p-p0
                # print('Stable equilibrium found after %d iterations, tolerance = %1.4e\n', i,tolerance)
                break
             
            p0=p
        
        return p

    def FindUnstableEqb(self, l, u, fx):
        # Procedure to find unstable equilibrium using bisections between l and u 
        # inputs 
        #   l,u: lower and upper bounds on initial interval
        #   u: upper bound on initial interval
        #   fx: second order best response
        # output
        #   p: unstable equilibrium 
        #   NOTE in case of multiple fixed points, p depends which p0 initializes the algorithm. 

        m=(l+u)/2 
        fm=fx(m)
        fl=fx(l)
        fu=fx(u)
        convergence=0          
        p=np.nan
        for iter in range(self.max_sa):
            if fm>=m:
                u=m
            elif fm<m:
                l=m
            elif fm==m:
                l=m
                u=m
            
            m=(l+u)/2
            fm=fx(m)
            # fprintf('%d new [l,u] interval is [%g,%g]\n',i,l,u)
            tolerance=abs(l-m)
            if tolerance<1e-6:
                p=m 
                convergence=1          
                # fprintf('Unstable equilibrium found after %d iterations, tolerance = %1.4e\n', i,tolerance)
                break
            
        if convergence==0:
            # fprintf('%d new [l,u] interval is [%g,%g]\n',i,l,u)
            print('FindUnstableEqb did not converge')

        return p

    def simdata(self):
        randnum=np.random.rand(self.N, 2)      

        # Solve for equilibrium probabilities
        pa_all=self.FindEqb(self.br2_a)
        pb_all=self.br_b(pa_all)
        neqb=np.size(pa_all)
        esr=min(self.esr,neqb-1)
        p_a=pa_all[esr];
        p_b=self.br_b(p_a);

        print('pa:', p_a)
        print('pb:', p_b)

        dta=np.ones([self.N , 2])
        dta[:,0]=1*(randnum[:,0]<p_a)
        dta[:,1]=1*(randnum[:,1]<p_b)
        return dta

    def logl(self, model, data, theta):
        # sgame.logl: log likelihood function for NFXP estimation static entry game
        d_a=data[:,0]
        d_b=data[:,1]

        # model=self.update(model, theta); 

        model.alpha=theta[0];
        model.beta=theta[1];
        pa=model.FindEqb(model.br2_a)
        pb=model.br_b(pa)
        neqb=np.size(pa)  

        logl_ieqb=np.ones([neqb,1]);
        # compute log likelihood associated with each equilibrium
        for ieqb in range(neqb):
            logl_i= d_a*np.log(pa[ieqb]) + (1-d_a)*np.log(1-pa[ieqb]) \
                  + d_b*np.log(pb[ieqb]) + (1-d_b)*np.log(1-pb[ieqb])  
            logl_ieqb[ieqb,:]=sum(logl_i)
        logl=max(logl_ieqb)
        return logl

    def plt_br(self, fname='br.png'):
        # plot(br_a(pvec), pvec, '-r')
        # hold on
        # plot(pvec, br_b(pvec), '-b')
        # hold on;
        # plot(pa, pb, 'sk')
        # strValues = strtrim(cellstr(num2str([pa(:) pb(:)],'(%1.3f,%1.3f)')));
        # text(pa+ 0.03,pb + 0.03,strValues,'VerticalAlignment','bottom');
        # legend('\psi_a(p_b)','\psi_b(p_a)')

        # Plot best response functions
        pvec=np.arange(0, 1, 0.001);
        fig1, ax = plt.subplots(1,1,figsize=(8,8))
        ax.set_title('Best response functions, firm a and firm b')
        ax.plot(self.br_a(pvec), pvec, '-r', label=r'$\psi_a(p_b)$');
        ax.plot(pvec, self.br_b(pvec), '-b', label=r'$\psi_b(p_a)$');

        # Solve for equilibrium probabilities
        pa=self.FindEqb(self.br2_a)
        pb=self.br_b(pa)

        ax.plot(pa, pb, 'sk')
        # strValues = strtrim(cellstr(num2str([pa(:)],'(%1.3f)')));
        # text(pa + 0.03,pa - 0.03,strValues,'VerticalAlignment','bottom');
        ax.legend()
        ax.set_xlabel('$p_a$')
        ax.set_ylabel('$p_b$')
        ax.set_xlim(0.0, 1)
        ax.set_ylim(0.0, 1)
        for xy in zip(pa, pb):                                      
            ax.annotate('      (%5.3f, %5.3f)' % xy, xy=xy, textcoords='data') 
        plt.savefig(fname='br.png', dpi=150)

    def plt_2br(self, fname='br.png'):
        # Plot second order best response functions
        pvec=np.arange(0, 1, 0.001);
        fig1, ax = plt.subplots(1,1,figsize=(8,8))
        ax.set_title('Second order best response function, firm a')
        ax.plot(pvec, self.br2_a(pvec),'r', label=r'${\psi_a(\psi_b(p_a))}$');
        ax.plot( pvec, pvec, 'k-');

        # Solve for equilibrium probabilities
        pa=self.FindEqb(self.br2_a)


        ax.plot(pa, pa, 'sk')
        # strValues = strtrim(cellstr(num2str([pa(:)],'(%1.3f)')));
        # text(pa + 0.03,pa - 0.03,strValues,'VerticalAlignment','bottom');
        ax.legend()
        ax.set_xlabel('$p_a$')
        ax.set_ylabel('$p_b$')
        ax.set_xlim(0.0, 1)
        ax.set_ylim(0.0, 1)
        for xy in zip(pa, pa):                                      
            ax.annotate('      (%5.3f, %5.3f)' % xy, xy=xy, textcoords='data') 
        plt.savefig(fname='br.png', dpi=150)

        
