clear;
close all;
global output;

% Set parameters
alpha=5;
beta=-11;
x_a=0.52;
x_b=0.22;
T=10000; % number of time game is played
esr=1;  % Eqiulibrium to be selected

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

% Simulate data
[d_a,d_b, eqbinfor]= sgame.simdata(x_a, x_b, alpha, beta, esr, rand(T,2));

% stating values
phat_a(1)=mean(d_a);
phat_b(1)=mean(d_b);
% phat_a=1; phat_b=1;

theta0=[alpha; beta]; % at true parameters
theta0_mpec=[theta0; phat_a; phat_b];

%mle
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

%mpec
% Define objective functions and constraints
ll_p_mpec = @(theta) sgame.logl_mpec(d_a,d_b, x_a, x_b, theta(1), theta(2), theta(3), theta(4)); % Objective function: negative of log likelihood
con_p_BNequations = @(theta) sgame.con_BNequations(x_a,x_b, theta(1), theta(2), theta(3), theta(4)); % Constraint (Bayesian-Nash Equilibrium equations) 	



% Upper and lower bounds on parameters
lb = zeros(4,1); ub = zeros(4,1);
lb(1) = -inf; ub(1) = inf; %alpha
lb(2) = -inf; ub(2) = inf; %beta
lb(3) = 0; ub(3) = 1; %p_a
lb(4) = 0; ub(4) = 1; 	%p_b

lb(1) = 4; ub(1) = 6; 		%alpha
lb(2) = -12; ub(2) = -10; %beta
lb(3) = 0; ub(3) = .1; 		%p_a
lb(4) = 0.7; ub(4) = .8;	%p_b

% No linear equality constraints
Aeq = []; beq = [];

options_mpec = optimset('Display','iter',...
'GradConstr','off','GradObj','off','TolCon',1E-6,'TolFun',1E-10,'TolX',1E-15,'MaxFunEval', 100000 ); 
options_mpec = optimset('Display','iter','GradConstr','off','GradObj','off','TolCon',1E-6,'TolFun',1E-10,'TolX',1E-15,'MaxFunEval', 100000, 'Algorithm','interior-point' ); 
outsidetimer = tic;

[theta_hat_mpec,FVAL,EXITFLAG,OUTPUT2] = fmincon(ll_p_mpec,theta0_mpec,[],[],Aeq,beq,lb,ub,con_p_BNequations,options_mpec);  
timetoestimate = toc(outsidetimer)
				
	
theta_hat_mpec

% Compute likelihood over a grid of alpha and p_a at true values of beta and p_b
ll_p_mpec_con = @(theta) ll_p_mpec([theta(1); beta; theta(2); pb(esr)]);
con_p_mpec = @(theta) con_p_BNequations([theta(1); beta; theta(2); pb(esr)]);
alphavec=alpha-5:0.5:alpha+5;
pa_vec=max(pa(esr)-0.2,0):0.002:min(pa(esr)+0.05, 1);
% pa_vec=0:0.01:1;
[alphagrid, pagrid]=meshgrid(alphavec, pa_vec);
	for r=1:numel(pa_vec);
		for c=1:numel(alphavec);
			llfig(r,c)=ll_p_mpec_con([alphavec(c), pa_vec(r)]);
			[~,ceq]=con_p_mpec([alphavec(c), pa_vec(r)]);
			con1(r,c)=ceq(1);
			con2(r,c)=ceq(2);
		end
	end

	% Maximize likelihood function. 
	% Note: fminsearch is based on the Nelder-Mead algorithm, that does not rely on objective function being differentiable.
	%[theta_hat, FVAL]=fminsearch(@(theta)-logl(theta), [alpha, beta])
	figure(1)
	subplot(2,2,1), surfc(alphagrid, pagrid,llfig)
	colormap jet  
	title(sprintf('Log-likelihood function, Equilbrium %d played on the data', esr));
	%hold on
	%plot3(theta_hat(1), theta_hat(2), -FVAL,'Marker', 's', 'MarkerSize',15);
	xlabel('Alpha')  
	ylabel('p_a')  

	subplot(2,2,2), surfc(alphagrid, pagrid,con1)
	colormap jet  
	title(sprintf('BNE constraint 1, Equilbrium %d played on the data', esr));
	%hold on
	%plot3(theta_hat(1), theta_hat(2), -FVAL,'Marker', 's', 'MarkerSize',15);
	xlabel('Alpha')  
	ylabel('p_a')  

	subplot(2,2,3), surfc(alphagrid, pagrid,con2)
	colormap jet  
	title(sprintf('BNE constraint 2, Equilbrium %d played on the data', esr));
	%hold on
	%plot3(theta_hat(1), theta_hat(2), -FVAL,'Marker', 's', 'MarkerSize',15);
	xlabel('Alpha')  
	ylabel('p_a')  
