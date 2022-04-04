clear;
close all;
global output p_a_nfxp p_b_nfxp neqb_nfxp;


% Set parameters
alpha=5;			% Monopoly profit (if x_i=1, i=a,b)	
beta=-11;			% Duopoly profit (if x_i=1, i=a,b)
T=625; 				% number of time game is played
M=2; 				% number of markets

estimator='mle-mpec';
estimator='mle-nfxp';
% Firm types
ma=5; 				% number firm a types
mb=5;					% number firm b types

nMC=100;

% grid over types; 
xa_grid=linspace(0.12, 0.87, ma)'
xb_grid=linspace(0.12, 0.87, mb)'

pa_sel=nan(nMC,M);
pb_sel=nan(nMC,M);
pa_hat=nan(nMC,M);
pb_hat=nan(nMC,M);
converged=nan(nMC, 1);
neqb=nan(nMC, M);

% Equilibrium selection for DGP
sw_esr= 3; 	% if sw_esr= 1,2, or 3 equilibrium 1,2, or 3 is played in the data (for all markets). 
					 	% if sw_esr= 123, equilibrium 1,2,and 3 is selected randomly at each market. 
					 	% if sw_esr= 13,  equilibrium 1 or 3 are selected randomly at each market. 

% Below we will generate data from multiple equilibria, according to an equilirbrium selection rule esr for each market.
% The equilibrium selection rule is a number between 1 and 3. If esr>1 and there is only a unique equilibrium (neqb=1), 
% then the first feasible equilibrium is picked, i.e. min(neqb,esr). 
% Equilibrium are sorted after ascending in p_a 
theta0=1*[alpha; beta];
% Begin Monte Carlo Experiment
for iMC=1:nMC; % Loop over equilibria
	fprintf('Monte Carlo replication iMc=%d out of nMC=%d\n', iMC, nMC);


	
	% Equilibrium selection in the data.
	if sw_esr <=3; 
		esr=ones(M, 1)*sw_esr;	% Always play equilibrium sw_esr at each market
	elseif sw_esr ==123; 
		esr=ceil(rand(M,1)*3);  % Equilibrium 1,2 or 3 to be selected randomly
	elseif sw_esr ==13; 
		u=rand(M,1);
		esr=(u<0.5)*1 + (u>=0.5)*3;  % Equilibrium 1 or 3 to be selected random
	end

	% Simulate data
	if 1
		u=rand(M,2); 
		x_a=xa_grid(ceil(u(:,1)*ma));
		x_b=xa_grid(ceil(u(:,2)*mb));
	else
		x_a=kron(ones(mb,1),linspace(0.12, 0.87, ma)');
		x_b=kron(linspace(0.12, 0.87, mb)', ones(ma,1));
		M=numel(x_a);	% number of markets 
	end;


	[d_a,d_b, eqbinfo]= sgame.simdata_panel(x_a, x_b, alpha, beta, esr, T);

	for im=1:M;

		% Best response functions, 
		br_a = @(p_b) sgame.br_a(p_b,x_a(im), alpha, beta); % firm a
		br_b = @(p_a) sgame.br_b(p_a,x_b(im), alpha, beta); % firm a
		br2_a = @(p_a) sgame.br2_a(p_a,x_a(im),x_b(im), alpha, beta); % firm a

		% % Solve for equilibrium probabilities
		pa_true=sgame.FindEqb(0,1, br2_a);
		pb_true=br_b(pa_true);
		neqb(iMC, im)=numel(pa_true);
		esr(im)=min(neqb(iMC, im),esr(im));
		pa_sel(iMC, im)=pa_true(esr(im));
		pb_sel(iMC, im)=pb_true(esr(im));
	end


	% first step estimates (stating values for mpec and npl) 
	phat_a=mean(d_a)';
	phat_b=mean(d_b)';

  if strcmp(estimator,'mle-nfxp');
  	logl=@(theta) sgame.logl_panel(d_a,d_b, x_a, x_b, theta(1), theta(2));

		% Maximize likelihood function (NFXP approach). 
		% Note: fminsearch is based on the Nelder-Mead algorithm, that does not rely on objective function being differentiable.
		[theta_hat(iMC,:), FVAL(iMC), converged(iMC)]=fminsearch(@(theta)-logl(theta), theta0);
		pa_hat(iMC, :)=p_a_nfxp;
		pb_hat(iMC, :)=p_b_nfxp;
	elseif strcmp(estimator,'mle-mpec');
				% Define objective functions and constraints
		theta0_mpec=[theta0; phat_a; phat_b];
		ll_p_mpec = @(theta) sgame.logl_panel_mpec(d_a,d_b, x_a, x_b, theta); % Objective function: negative of log likelihood
		con_p_BNequations = @(theta) sgame.con_BNequations_panel(x_a,x_b, theta); % Constraint (Bayesian-Nash Equilibrium equations) 	

		% Upper and lower bounds on parameters
		lb = zeros(2+2*M,1); ub = zeros(2+2*M,1);
		lb(1) = -inf; ub(1) = inf; %alpha
		lb(2) = -inf; ub(2) = inf; %beta
		lb(3:end) = 0; ub(3:end) = 1; %p_a

		lb(1) = 0; ub(1) = 20; 		%alpha
		lb(2) = -20; ub(2) = 20; %beta

		% No linear equality constraints
		Aeq = []; beq = [];

		options_mpec = optimset('Display','off','GradConstr','off','GradObj','off','TolCon',1E-5,'TolFun',1E-4,'TolX',1E-15,'MaxFunEval', 10000, 'Algorithm','interior-point' ); 
		outsidetimer = tic;
		[theta_hat(iMC,:), FVAL(iMC), EXITFLAG,OUTPUT2] = fmincon(ll_p_mpec,theta0_mpec,[],[],Aeq,beq,lb,ub,con_p_BNequations,options_mpec);  
		converged(iMC)=(EXITFLAG==1);
		timetoestimate = toc(outsidetimer);

		pa_hat(iMC, :)=theta_hat(iMC,3:M+2);
		pb_hat(iMC, :)=theta_hat(iMC,M+3:end);
	end
end

% FIGURE : Plot distribution of motecarlo results when data is generated from equilibrium 1,2 or 3
figure(1); 
nbar=60;
subplot(2,1,1), hist(theta_hat(:,1),nbar); title(sprintf('True alpha=%1.1f', alpha));  
subplot(2,1,2), hist(theta_hat(:,2),nbar); title(sprintf('True beta=%1.1f', beta));  

	figure(2); 
	subplot(2,1,1), hist(reshape(pa_sel(converged==1,:)-pa_hat(converged==1,:),numel(pa_hat(converged==1,:)),1), 60);
	title('p_a: true selected - estimated')
	subplot(2,1,2), hist(reshape(pb_sel(converged==1,:)-pb_hat(converged==1,:),numel(pb_hat(converged==1,:)),1),60)
	title('p_b: true selected - estimated')
