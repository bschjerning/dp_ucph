clear all;
% close all;
global output;

% Set parameters
alpha=5;
beta=-11;
x_a=0.52;
x_b=0.22;
T=50 % number of times game is played

for esr=1:3;  % loop over eqiulibrium to be selected in the data
fprintf('Data generated from equilirbrium %d\n', esr);

% esr is Equilibrium selection rule (number between 1 and 3). 
% If ESR>1 and there is only a unique equilibrium (neqb=1), 
% then the first feasible equilibrium is picked, i.e. min(neqb,esr).
% Equilibria are sorted after ascending in p_a 

nMC=1000;  % number of MC reps

estimator='pml2step';
estimator='npl';
% estimator='mle';
% estimator='mpec';

% Best response functions, 
br_a = @(p_b) sgame.br_a(p_b,x_a, alpha, beta); % firm a
br_b = @(p_a) sgame.br_b(p_a,x_b, alpha, beta); % firm a
br2_a = @(p_a) sgame.br2_a(p_a,x_a,x_b, alpha, beta); % firm a

% % Solve for equilibrium probabilities
pa_true=sgame.FindEqb(0,1, br2_a);
pb_true=br_b(pa_true);

pa_hat=nan(nMC, 3);
pb_hat=nan(nMC, 3);
pa_sel=nan(nMC,1);
pb_sel=nan(nMC,1);
converged=zeros(nMC, 1);

for iMC=1:nMC; % Loop over equilibria
	fprintf('Monte Carlo replication iMc=%d out of nMC=%d\n', iMC, nMC);
	% Simulate data
	[d_a,d_b, eqbinfor]= sgame.simdata(x_a, x_b, alpha, beta, esr, rand(T,2));

	% stating values
	phat_a(1)=mean(d_a);
	phat_b(1)=mean(d_b);
	% phat_a=1; phat_b=1;

	theta0=[alpha; beta]; % at true parameters
	if strcmp(estimator,'mle');
  	logl=@(theta) sgame.logl(d_a,d_b, x_a, x_b, theta(1), theta(2));
  	[theta_hat(iMC,:), FVAL(iMC), converged(iMC)]=fminsearch(@(theta)-logl(theta), theta0); % Maximize likelihood function. 
	
	elseif strcmp(estimator,'pml2step');
    	logl=@(theta) sgame.logl_pml2step(d_a,d_b, x_a, x_b, theta(1), theta(2), phat_a, phat_b);
    	options= optimset('Display','off');
    	[theta_hat(iMC,:), FVAL(iMC), converged(iMC)]=fminunc(@(theta)-logl(theta), theta0, options); 	% Maximize likelihood function. 
	
	elseif strcmp(estimator,'npl');
		K=100;
		x=nan(K,2);
		phat0=[phat_a(1),phat_b(1)];
		for k=1:K;		% step 1: estimate phat0
			[x(k+1,:), phat1, fval]=sgame.npl_step(d_a,d_b, x_a, x_b, phat0, theta0);
				tolerance = max(abs(x(k+1,:)-x(k,:)));
			phat0=phat1; 
			theta0=x(k+1,:);
			if tolerance<0.05;
	    	converged(iMC)=1;
				break
			end
		end

    	theta_hat(iMC,:)=x(k+1,:	);
    	FVAL(iMC)=fval;

	elseif strcmp(estimator,'mpec')
	  theta0_mpec=0*[theta0; phat_a; phat_b];

		% Define objective functions and constraints
		ll_p_mpec = @(theta) sgame.logl_mpec(d_a,d_b, x_a, x_b, theta(1), theta(2), theta(3), theta(4)); % Objective function: negative of log likelihood
		con_p_BNequations = @(theta) sgame.con_BNequations(x_a,x_b, theta(1), theta(2), theta(3), theta(4)); % Constraint (Bayesian-Nash Equilibrium equations) 	

		% Upper and lower bounds on parameters
		lb = zeros(4,1); ub = zeros(4,1);
		lb(1) = -inf; ub(1) = inf; %alpha
		lb(2) = -inf; ub(2) = inf; %beta
		lb(3) = 0; ub(3) = 1; %p_a
		lb(4) = 0; ub(4) = 1; 	%p_b

		lb(1) = 0; ub(1) = 20; 		%alpha
		lb(2) = -20; ub(2) = 20; %beta
		lb(3) = 0; ub(3) = 1; 		%p_a
		lb(4) = 0; ub(4) = 1;	%p_b

		% No linear equality constraints
		Aeq = []; beq = [];

		options_mpec = optimset('Display','off','GradConstr','off','GradObj','off','TolCon',1E-5,'TolFun',1E-4,'TolX',1E-15,'MaxFunEval', 100000, 'Algorithm','interior-point' ); 
		outsidetimer = tic;
		[theta_hat(iMC,:), FVAL(iMC), EXITFLAG,OUTPUT2] = fmincon(ll_p_mpec,theta0_mpec,[],[],Aeq,beq,lb,ub,con_p_BNequations,options_mpec);  
		converged(iMC)=(EXITFLAG==1);
		timetoestimate = toc(outsidetimer);
				
  end
	
	% % Second order best response function at estimated parameters
	br2_a = @(p_a) sgame.br2_a(p_a,x_a,x_b, theta_hat(iMC,1), theta_hat(iMC, 2)); % firm a

	% % Solve for equilibrium probabilities
	pa=sgame.FindEqb(0,1, br2_a);
	pb=br_b(pa);

	neqb=numel(pa);	
	% if converged(iMC);
		pa_hat(iMC,1:neqb)=pa';
		pb_hat(iMC,1:neqb)=pb';
		pa_sel(iMC,1)=pa(min(esr, neqb));
		pb_sel(iMC,1)=pb(min(esr, neqb));

		if strcmp(estimator,'mpec');
			pa_sel(iMC,1)=theta_hat(iMC, 3);
			pb_sel(iMC,1)=theta_hat(iMC, 4);
		end
	% end
end

figure(esr); 
nbar=20;
subplot(2,2,1), hist(theta_hat(:,1),nbar); title(sprintf('True alpha=%1.1f', alpha));  %xlim([4 6]);
subplot(2,2,2), hist(theta_hat(:,2),nbar); title(sprintf('True beta=%1.1f', beta));  %xlim([-13 -9]);
subplot(2,2,3), hist(pa_sel,nbar); title(sprintf('True p_a=%1.3f', pa_true(esr)));  
subplot(2,2,4), hist(pb_sel,nbar); title(sprintf('True p_b=%1.3f', pb_true(esr)));  
sgtitle(sprintf('Estimator: %s, equilbrium %d',estimator, esr))
fprintf('\n\n')
end