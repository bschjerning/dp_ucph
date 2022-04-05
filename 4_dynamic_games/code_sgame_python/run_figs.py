import numpy as np
from sgame import *
import matplotlib.pyplot as plt

# create instance of sgame model object
model=sgame(alpha=5, beta=-11, x_a=0.01, x_b=0.01)

# Solve for equilibrium probabilities
pa=model.FindEqb(model.br2_a)
pb=model.br_b(pa)
print('Equilibrium investmet probabilities, firm a\n', pa, '\n')
print('Equilibrium investmet probabilities, firm b\n', pb)

# Plot best response functions for firm a and b
model.plt_br()

# Plot second order best response function for firm a
model.plt_2br()

plt.show()

	















quit()





# simulate data
model.esr=1
model.N=10000
data=model.simdata()
print('sum(data,1)', np.sum(data,axis=0))

theta=[model.alpha, model.beta]
ll=model.logl(model, data, theta)
print('ll=',ll,'\n')
print(ll)

ngrid=[3,3]
alphavec=np.linspace(model.alpha-5, model.alpha+5, ngrid[0]);
betavec=np.linspace(model.beta-5, model.beta+5, ngrid[1]);
print(alphavec, betavec);


ll=np.ones(ngrid)
print(ll)
ialpha=0
for alpha in np.linspace(model.alpha-5, model.alpha+5, ngrid[0]):
	ibeta=0;
	for beta in np.linspace(model.beta-5, model.beta+5, ngrid[1]):
		print('alpha=',alpha, 'beta=', beta, 'ialpha=',ialpha, 'ibeta=', ibeta)
		ll[ialpha, ibeta]=model.logl(model, data, [alpha, beta])

		ibeta+=1;
	ialpha+=1;
print(ll)

quit()

# function [d_a,d_b, eqbinfo]=simdata(x_a, x_b, alpha, beta, esr, randnum)

N=100
randnum=np.random.rand(N, 2)
esr=2

# Solve for equilibrium probabilities
pa_all=model.FindEqb(model.br2_a)
pb_all=br_b(pa_all)
neqb=np.size(pa_all)
esr=min(esr,neqb-1)
p_a=pa_all[esr]
p_b=model.br_b(p_a)
d_a=1*(randnum[:,0]<p_a)
d_b=1*(randnum[:,1]<p_b)
print(d_a)
print(d_b)








# plot(pa, pa, 'sk')
# strValues = strtrim(cellstr(num2str([pa(:)],'(%1.3f)')));
# text(pa + 0.03,pa - 0.03,strValues,'VerticalAlignment','bottom');
# title('Second order best response function, firm a')
# legend('\psi_a(\psi_b(p_a))')
# xlabel('p_a')
# ylabel('p_b')


