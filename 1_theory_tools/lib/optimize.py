import numpy as np
import scipy
import time
def print_out(iter, x, dx, fx, gx, step, *kwargs):
    if iter==0:
        print('Starting values: ', x[:].T)

        print('%4s %10s  %10s  %10s   %10s'%('iter','step', '||dx||','||g(x)||', 'f(x)'))
        print('-'*100)

    print('%4d  %10.2g  %10.2e  %10.2e  %10.5g'%(iter,step, np.max(np.abs(dx)),max(np.abs(gx)),fx))

def newton(f, x0, user_grad=0, max_iter=100, gtol=1e-8, callback=print_out, use_derivatives=1, linesearch =1, maximize=1): 
      '''Minimize f(x) using Newtons method.'''

      tic = time.perf_counter()
    
      x = np.array(x0)
      f0=f(x) 
      convergence=0
      step=1;

      for iter in range(max_iter):
            if use_derivatives==1:
                  fx, gx, Hx=f(x, use_derivatives)
            else:
                  fx=f(x)
                  gx=gradient(f, x)
                  Hx=hessian(f, x)

            # Newton step
            dx = np.linalg.solve(Hx,gx)
            dx.shape=(len(x),1);

            # compute optimal step length. 
            if maximize==1:
                  stepfun =  lambda step: -f(x[:]-step*dx)
            else: 
                  stepfun =  lambda step: f(x[:]-step*dx)

            if linesearch:
                  step = scipy.optimize.brute(stepfun,(slice(0.05, 2, 0.05),),finish=None)
                  # res=scipy.optimize.minimize_scalar(stepfun, bounds=(0.05, 2), method='golden')
                  # step=res.x
            x[:]=x[:]-step*dx
            
            if callback:
                  callback(iter, x, dx, fx, gx, step)
            if max(abs(gx)) < gtol:
                  convergence=1
                  print ('Newton converged after %d iterations, ||g(x)|| = %1.4e' % (1+iter,max(gx)))
                  toc = time.perf_counter()
                  print(f"Elapsed time: {toc - tic:0.4f} seconds")

                  return x
      print ('Newton failed to converge after %d iterations, ||g(x)|| = %1.4e' % (1+iter,max(gx)))

      return None

def gradient(f, x0, delta=1e-4):
	g = np.zeros((len(x0)), dtype = np.float)
	for i in range(len(x0)):
		x1 = x0.copy()
		x2 = x0.copy()
		x1[i] += delta
		x2[i] -= delta
		g[i] = (f(x1) - f(x2)) / (2 * delta)
	return g

def hessian(f, x0, delta=1e-8):
	H = np.zeros([len(x0), len(x0)], dtype = np.float)

	for i in range(len(x0)):
		x1 = x0.copy()
		x2 = x0.copy()

		x1[i] += delta
		x2[i] -= delta

		H[:,i] = (gradient(f, x1) - gradient(f, x2)) / (2 * delta)

	return H
