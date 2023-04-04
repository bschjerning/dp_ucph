clear;
close all;
global output;

% Set parameters
% alpha*x_j: monopoly profits for firm j
% beta*x_j:  duopoly profits for firm j

% multible equilibria at the the true value of parameters
alpha=5; beta=-11; x_a=0.52; x_b=0.22; 
% beta=-1; % unique uequilibrium
% x_a=.5; x_b=.5; symetric firms
sigma=1; % scale iid. extreme value error

x_a=x_a/sigma;
x_b=x_b/sigma;

N=1000; % number of time game is played (number of observations in the data)

% Best response functions, 
br_a = @(p_b) sgame.br_a(p_b,x_a, alpha, beta); % firm a
br_b = @(p_a) sgame.br_b(p_a,x_b, alpha, beta); % firm a

% Second order best response function
br2_a = @(p_a) sgame.br2_a(p_a,x_a,x_b, alpha, beta); % firm a

% Solve for equilibrium probabilities
pa=sgame.FindEqb(0,1, br2_a);
pb=br_b(pa);

% FIGURE 1: Plot second order best response functions
figure(1)
pvec=(0:0.01:1)';
plot(pvec,[br2_a(pvec) pvec], 'LineWidth', 3);
hold on
plot(pa, pa, 'sk', 'LineWidth', 4);
strValues = strtrim(cellstr(num2str([pa(:)],'(%1.3f)')));
text(pa + 0.03,pa - 0.03,strValues,'VerticalAlignment','bottom');
title('Second order best response function, firm a');
legend('\psi_a(\psi_b(p_a))');
xlabel('p_a');
ylabel('p_a');
fontsize(gcf,scale=1.5);

% FIGURE 2: Plot best response functions for firm a and firm b
figure(2)
plot(br_a(pvec), pvec, '-r', 'LineWidth', 3)
hold on
plot(pvec, br_b(pvec), '-b', 'LineWidth', 3)
hold on;
plot(pa, pb, 'sk', 'LineWidth', 4)
strValues = strtrim(cellstr(num2str([pa(:) pb(:)],'(%1.3f,%1.3f)')));
text(pa+ 0.03,pb + 0.03,strValues,'VerticalAlignment','bottom');
title('Best response functions, firm a and firm b');
legend('\psi_a(p_b)','\psi_b(p_a)');
xlabel('p_a');
ylabel('p_b');
fontsize(gcf,scale=1.5);

% FIGURE 3-5: Plot likelihood functions when data is generated from equilibrium 1,2 or 3
% Below we loop over esr=1:3, where esr is Equilibrium selection rule (number between 1 and 3). 
% If ESR>1 and there is only a unique equilibrium (neqb=1), 
% then the first feasible equilibrium is picked, i.e. min(neqb,esr).
% Equilibria are sorted after ascending in p_a 

for esr=1:numel(pa); % Loop over equilibria
	% Simulate data
	[d_a,d_b, eqbinfor]= sgame.simdata(x_a, x_b, alpha, beta, esr, rand(N,2));

	% Define likelihood function
	logl=@(theta) sgame.logl(d_a,d_b, x_a, x_b, theta(1), theta(2));

	% Compute likelihood over a grid of alpha and beta
	alphavec=alpha-5:0.25:alpha+5;
	betavec=beta-5:0.25:beta+5;
	[alphagrid, betagrid]=meshgrid(alphavec, betavec);
	for r=1:numel(betavec);
		for c=1:numel(alphavec);
			llfig(r,c)=logl([alphavec(c), betavec(r)]);
		end
	end

	% Maximize likelihood function. 
	% Note: fminsearch is based on the Nelder-Mead algorithm, that does not rely on objective function being differentiable.
	[theta_hat, FVAL]=fminsearch(@(theta)-logl(theta), [alpha, beta]);
	figure(2+esr) 
	surfc(alphagrid, betagrid,llfig)
	colormap jet  
	title(sprintf('Log-likelihood function, Equilbrium %d played on the data', esr));
	hold on
	plot3(theta_hat(1), theta_hat(2), -FVAL,'Marker', 'o', 'MarkerSize',15, 'MarkerEdgeColor',[0 0 1], 'MarkerFaceColor',[0 0 1]);
	hold on
	plot3(alpha,beta, logl([alpha, beta]),'Marker', 'o', 'MarkerSize',15, 'MarkerEdgeColor',[0 0 1], 'MarkerFaceColor',[1 1 1]);
	xlabel('Alpha')  
	ylabel('Beta') 
	legend('Likelihood', 'contour', 'MLE',  'true value');
	fontsize(gcf,scale=1.2)

end

