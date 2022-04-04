% This program solves and the social planners of optimal cost reducing investments in 
% Iskhakov Rust and Schjerning (2015): The Dynamics of Bertrand Price Competition with Cost-Reducing Investments
% By Bertel Schjerning

%**************************************************************************
% 1: Setup  
%**************************************************************************

% clear all
clc
clear mp;

do_VFI=0; 

% Model parameters
mp.k0=0;
mp.k1=5;
mp.N=200;
mp.beta=0.99;
% algorithm parameters
ap.printfxp=1; % only print final itaration info for fixed point algorithm
ap.sa_max =1000;

% update paramaters and set un-initialized parameters to default
mp=spp.setup(mp);
ap=solveDP.setup(ap);

%**************************************************************************
% 1: Solve social planners problem by state recursion  
%**************************************************************************

tic
fprintf('\nSolve social planners problem solved state recursion algorithm\n');
[V, P] = spp.state_recursion(mp);
time_SR=toc;
fprintf('Social planners problem solved in %g seconds using state recursion algorithm\n', time_SR);
%**************************************************************************
% 2: Solve social planners problem by successive approximations
%**************************************************************************

if do_VFI
	tic
	fprintf('\nSolve social planners problem solved VFI\n');
	[V_VFI, P_VFI]=solveDP.sa(@(V) spp.bellman(V, mp), 0, ap);
	time_VFI=toc;
	fprintf('Social planners problem solved in %g seconds using VFI\n', time_VFI);
end
%**************************************************************************
% 3: Simulate investment dynamics
%**************************************************************************
   
[sim]=spp.simulate(P, mp);

%**************************************************************************
% 4: Graph Results
%**************************************************************************

h1 = figure(1);
surf(mp.c,mp.c,P, V)
surf(mp.c,mp.c,P)
title('Policy function, solcial planner')
ylabel('Current marginal cost, x') % Corresponds to rows in V
xlabel('State of the art marginal cost, c');  % Corresponds to columns in V
view(2);

h2=figure(2);
surf(mp.c,mp.c,V)
title('Value function, solcial planner')
ylabel('Current marginal cost, x') % Corresponds to rows in V
xlabel('State of the art marginal cost, c');  % Corresponds to columns in V
zlabel('Value function')
view([-45 -10])

h3=figure(3);
stairs(sim.t, [sim.x sim.c], 'Linewidth',2);    
title('Investment dynamics, solcial planner')
xlabel('Time, t');  % Corresponds to columns in V
ylabel('Marginal cost') % Corresponds to rows in V
legend('Current cost, x','State of the art cost, c')
xlim([0 max(sim.t)])

