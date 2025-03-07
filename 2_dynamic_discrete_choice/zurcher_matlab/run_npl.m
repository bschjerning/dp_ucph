% Strutural Estimation of Rust's engine replacement model using NPL 
close all
clear all
clc

% Swiches:
Kmax=10; 		% Max number of outer loop iterations for NPL 
bustypes = 4; 	% Select bus types smaller than this number: choose 3 or 4

% Set parameters (values for RC and mp.c will be used as starting values during estimation)
mp0.bellman_type='iv';  	% bellman in expected value ('ev') or ('iv') integrated value function space  
mp0.pnames_u={'RC', 'c'};	% utility parameters to be estimated
mp0.pnames_P={};         % set mp0.pnames_P={}; to skip estimation of transition parameters  
mp0.bustypes=[1,2,3,4];		% Vector with chosen bus types (elements can be 1,2,3,4) 

% Fill out remaining parameters and update parameter dependencies
mp=zurcher.setup(mp0);

% ************************************
% section 0: Load data 
% ************************************
data = zurcher.readbusdata(mp);
N=size(data.d,1); 

% ************************************
% section 1: ESTIMATE state transition matrix parameters, mp.p 
% ************************************
% Initialization: Estimate mp.p using frequency estimator 
tab = tabulate(data.dx1); % If you do not know what this does, print it :)
tab = tab(tab(:,3)>0,:);  % As above
mp.p = tab(1:end-1,3)/100; % As above
P   = zurcher.statetransition(mp); % Create the transition matrix based on pi_0, pi_1, pi_2, and pi_3

% ************************************
% section : INITIAL CCP's 
% ************************************
% Initial guess for the conditional choice probabilities

% stating values for RC and c
theta0 = 0*[9.7686;1.3428];  %  [mp.RC; mp.c]
pk_init=2;
switch pk_init
	case 1
		pk0=0.99; 
		fprintf('Initialize ccps with static p(x|keep)=%g\n', pk0)
		pk0 = ones(mp.n,1)*pk0;    % Starting values for CCPs
	case 2
		disp('Initialize ccps with flexible logit');
		deg=2; % degree of polynomial in flexible logit
		x=[ones(N,1) (data.x/mp.n).^1 (data.x/mp.n).^2 (data.x/mp.n).^3 (data.x/mp.n).^4 ]; 
		xg=[ones(mp.n,1) (mp.grid/mp.n).^1 (mp.grid/mp.n).^2 (mp.grid/mp.n).^3 (mp.grid/mp.n).^4];
		 
		options =  optimset('Algorithm','quasi-newton','Display','off');
		[theta_flex_logit, fval] = fminunc(@(theta) npl.ll_logit(theta, data.d, x(:,1:deg+1)) ,zeros(deg+1,1), options);
		pk0=1./(1+exp(xg(:,1:deg+1)*theta_flex_logit));
		fprintf('Specifiction: logit with %d degree polynomial in mileage \n',deg);
		fprintf('log-likelihood    = %10.3f \n',-N*fval);
	case 3
		disp('Initialize ccps with static enigne replacement model')
		% Use static logit model to intialize NPL (i.e. with beta=0 and Kmax=1)
		mp0=mp; mp0.beta=0;
		theta0 = 0*[9.7686;1.3428];
		[mp0, pk0, logl0, K0]=npl.estim(theta0, 0, data, P, mp0, 1);
		theta0=[mp0.RC, mp0.c];
	otherwise
		error('pk_init not valid');
end

% ************************************
% 2: Estimate Rust's model using NPL and NFXP
% ************************************
% NPL
[mp, pk, logl, K]=npl.estim(theta0, pk0, data, P, mp, Kmax);

% NFXP
ev0=0;
fprintf('\n*************************************************************************\n')
fprintf('Method: Nested Fixed point algorithm (NFXP)\n')
fprintf('*************************************************************************\n')

[nfxp_results, theta_hat, Avar]=nfxp.estim(data, mp);
mphat = output.estimates(mp, [mp.pnames_u mp.pnames_P], theta_hat, Avar);
fprintf('log-likelihood    = %10.3f \n',nfxp_results.llval);
fprintf('runtime (seconds) = %10.5f \n',nfxp_results.cputime);


% *******************************************
% 3: Solve estimated model using NPL and poly-algorithm  
% *******************************************

% Solve for fixed point of policy iteration operator, pk=Psi(pk)
fignr =1; % Figure number for figer that displays convergence of Policy iteration operator. (optional argument in npl.solve)
pk_npl_fixpoint  = npl.solve(mp, P, pk0, fignr);  % Solve for 

% Solve for fixed point of bellman operator in EV space, pk=Psi(pk)the model using the hybrid FXP algorithm, 
ap = dpsolver.setup; % Set default options

% bellman equation
bellman= @(ev) zurcher.bellman(ev, mp);

% solve using poly-algorithm (use combination of SA and NK)
[ev_fxp, pk_fxp]=dpsolver.poly(bellman, ev0, mp.ap, mp.beta);	

% Plot CCPs
figure(2)
hold all
plot(mp.grid,1-pk0,'-.b','LineWidth', 2);
plot(mp.grid,1-pk,'-k','LineWidth', 2);
plot(mp.grid,1-pk_npl_fixpoint,'-b','LineWidth', 2);
plot(mp.grid,1-pk_fxp,'--r','LineWidth', 2);
title(sprintf('Replacement probability, K=%d', K));
legend('Initial ccp','Last evaluation of \Psi','Fixed point of \Psi' ,'Fixed point of \Gamma','Location', 'southeast')
xlabel('Milage grid')
ylabel('Replacement probability')
grid on
ylim([0 0.16])

