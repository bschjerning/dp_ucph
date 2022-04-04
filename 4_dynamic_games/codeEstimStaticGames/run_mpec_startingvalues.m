clear;
close all;
global output p_a_nfxp p_b_nfxp neqb_nfxp k;

% Set parameters
alpha=5;
beta=-11;
x_a=0.52;
x_b=0.22;
T=1000; % number of time game is played
esr=1;  % Eqiulibrium to be selected in the simulated data

% starting values for equilirbrium probabilities
esr_start=0; % Choose equilibrium or set esr_start=0 to use frequency estimator as starting values;

% stating values of structural parameters
theta0=[alpha; beta]*0; % at true parameters

% esr is Equilibrium selection rule (number between 1 and 3). 
% If ESR>1 and there is only a unique equilibrium (neqb=1), 
% then the first feasible equilibrium is picked, i.e. min(neqb,esr).
% Equilibria are sorted after ascending in p_a 

% Best response functions, 
br_a = @(p_b) sgame.br_a(p_b,x_a, alpha, beta); % firm a
br_b = @(p_a) sgame.br_b(p_a,x_b, alpha, beta); % firm a
br2_a = @(p_a) sgame.br2_a(p_a,x_a,x_b, alpha, beta); % firm a

% % Solve for equilibrium probabilities
pa=sgame.FindEqb(0,1, br2_a);
pb=br_b(pa);

esr_start=min(numel(pa),esr_start);
esr=min(numel(pa),esr);

% Simulate data
[d_a,d_b, eqbinfor]= sgame.simdata(x_a, x_b, alpha, beta, esr, rand(T,2));


if esr_start==0
	%% start at frequency estimator
	phat_a(1)=mean(d_a);
	phat_b(1)=mean(d_b);

	% start at equilibrium esr_start
	theta0_mpec=[theta0; phat_a; phat_b];
else
	p0_a(1)=pa(esr_start);
	p0_b(1)=pb(esr_start);

	% Define objective functions and constraints
	ll_p_mpec0 = @(theta) sgame.logl_mpec(d_a,d_b, x_a, x_b, theta(1), theta(2), p0_a, p0_b); % Objective function: negative of log likelihood
	con_p_BNequations0 = @(theta) sgame.con_BNequations(x_a,x_b, theta(1), theta(2), p0_a, p0_b); % Constraint (Bayesian-Nash Equilibrium equations) 	

		% Upper and lower bounds on parameters
	lb = zeros(2,1); ub = zeros(2,1);
	lb(1) = -inf; ub(1) = inf; %alpha
	lb(2) = -inf; ub(2) = inf; %beta

	lb(1) = -100; ub(1) = 100; 		%alpha
	lb(2) = -100; ub(2) = 100; %beta

	% No linear equality constraints
	Aeq = []; beq = [];

	options_mpec = optimset('Display','iter','GradConstr','off','GradObj','off','TolCon',1E-6,'TolFun',1E-10,'TolX',1E-15,'MaxFunEval', 100000, 'Algorithm','interior-point' ); 
	outsidetimer = tic;

	[theta_hat_fixed_p,FVAL,EXITFLAG,OUTPUT2] = fmincon(ll_p_mpec0,theta0,[],[],Aeq,beq,lb,ub,con_p_BNequations0,options_mpec);  
	timetoestimate = toc(outsidetimer)
	theta0_mpec=[theta_hat_fixed_p; p0_a; p0_b];
end

%******************
%NFXP mle
%******************
logl=@(theta) sgame.logl(d_a,d_b, x_a, x_b, theta(1), theta(2));
[theta_hat_mle, FVAL_mle, converged_mle]=fminsearch(@(theta)-logl(theta), theta0); % Maximize likelihood function. 
theta_hat_mle

% % Second order best response function at estimated parameters
br2_a = @(p_a) sgame.br2_a(p_a,x_a,x_b, theta_hat_mle(1), theta_hat_mle(2)); % firm a

% % Solve for equilibrium probabilities
pa=sgame.FindEqb(0,1, br2_a);
pb=br_b(pa);

pa_hat=pa'
pb_hat=pb'

%******************
%mpec
%******************
% Define objective functions and constraints
ll_p_mpec = @(theta) sgame.logl_mpec(d_a,d_b, x_a, x_b, theta(1), theta(2), theta(3), theta(4)); % Objective function: negative of log likelihood
con_p_BNequations = @(theta) sgame.con_BNequations(x_a,x_b, theta(1), theta(2), theta(3), theta(4)); % Constraint (Bayesian-Nash Equilibrium equations) 	

% Upper and lower bounds on parameters
lb = zeros(4,1); ub = zeros(4,1);
lb(1) = -inf; ub(1) = inf; %alpha
lb(2) = -inf; ub(2) = inf; %beta
lb(3) = 0; ub(3) = 1; %p_a
lb(4) = 0; ub(4) = 1; 	%p_b

% No linear equality constraints
Aeq = []; beq = [];

options_mpec = optimset('Display','iter',...
'GradConstr','off','GradObj','off','TolCon',1E-6,'TolFun',1E-10,'TolX',1E-15,'MaxFunEval', 100000 ); 
options_mpec = optimset('Display','iter','GradConstr','off','GradObj','off','TolCon',1E-6,'TolFun',1E-10,'TolX',1E-15,'MaxFunEval', 100000, 'Algorithm','interior-point' ); 
outsidetimer = tic;

[theta_hat_mpec,FVAL_mpec,EXITFLAG,FMINCON_OUTPUT_MPEC] = fmincon(ll_p_mpec,theta0_mpec,[],[],Aeq,beq,lb,ub,con_p_BNequations,options_mpec);  
timetoestimate = toc(outsidetimer)
		
%******************
% PRINT RESULTS
%******************

fprintf('MLE-NFXP\n');	
fprintf('    %-16s %13s %13s %13s\n','Param.','True Value','Estimates','start val.');
pnames={'alpha','beta', 'pa', 'pb', 'esr'};
p0=[alpha; beta; pa(esr); pb(esr); esr];
est=[theta_hat_mle; p_a_nfxp; p_b_nfxp; k];
startv=[theta0; nan(3,1)];
FVAL= FVAL_mle;
fprintf('----------------------------------------------------------------------------------------\n');
for iP=1:numel(pnames);
  fprintf('    %-16s %13.4f %13.4f %13.4f \n', char(pnames(iP)),  p0(iP), est(iP), startv(iP));
end
fprintf('\n    %-16s %13.4f\n','likelihood', -FVAL);
fprintf('----------------------------------------------------------------------------------------\n');


fprintf('MLE-MPEC\n');	
fprintf('    %-16s %13s %13s %13s\n','Param.','True Value','Estimates','start val.');
pnames={'alpha','beta', 'pa', 'pb', 'esr'};
p0=[alpha; beta; pa(esr); pb(esr); esr];
est=[theta_hat_mpec; nan(1,1)];
startv=[theta0_mpec; esr_start];
FVAL= FVAL_mpec;
fprintf('----------------------------------------------------------------------------------------\n');
for iP=1:numel(pnames);
  fprintf('    %-16s %13.4f %13.4f %13.4f \n', char(pnames(iP)),  p0(iP), est(iP), startv(iP));
end
fprintf('\n    %-16s %13.4f\n','likelihood', -FVAL);
fprintf('----------------------------------------------------------------------------------------\n');
fprintf('\n    %-16s %13.4f\n','ll(MLE)-ll(MPEC)', -FVAL_mle-(-FVAL_mpec));
fprintf('----------------------------------------------------------------------------------------\n');
FMINCON_OUTPUT_MPEC
fprintf('----------------------------------------------------------------------------------------\n');
