classdef spp
  methods (Static)
	  function [mp] = setup(mpopt)
	  	% set default parameters
	  	mp.maxc= 5; % maximum state of the art cost
			mp.p = mp.maxc; % reservation price
			mp.N = 100; % Number of states in technological proces (c1,c2,c3,,c_[techlevels])
			mp.pf = 1; % Probability of technological improvement
			        % pi_h := Pr(c' = c_h+1 | c = c_h)
			mp.beta = exp(-0.05);% 0.9512 discount parameter: beta = exp(-r*dt)    
			mp.k0 = 0;   % investment cost parameter, constant
			mp.k1 = 8.3; % investment cost parameter, slope in K(c)=mp.k0+mp.k1/(1+c);
			mp.dt = 1; % 0.016; % timeincrement 
			mp.T =1000;  % Max number of periods to simulate

			% update paramaters with input
	    if nargin>0
	        pfields=fieldnames(mpopt);
	        for i=1:numel(pfields);
	            mp.(pfields{i})=mpopt.(pfields{i});
	        end
	    end

			% update dependent paramaters; 
			mp.c = linspace(mp.maxc,0,mp.N); % possible values of state of the art cost and social planners cost
			mp.prob = [ones(1,mp.N-1),0].*mp.pf; % transition probabilities for technological proces

	    % functional forms
	    mp.K=   @(c) mp.k0+mp.k1/(1+c);				% Investment cost as function of state of the art production cost c
	    mp.payoff= @(x) (mp.p-x).*mp.dt;  	 	% flow payoffs per unit time ... 
		end

		function [V,policy] = solve_last_corner(mp)
	    % Social planners productions cost
	    x = min(mp.c);
	    % State of the art production cost
	    c = x;
	    % Reservation price
	    p = mp.p;
	    % Discount rate
	    v_N = mp.payoff(x)/(1-mp.beta); % Value of not investing
	    v_I = v_N - mp.K(c); % Value of investing
	    V = max(v_I,v_N); % Value of value function in the state (x_K,c_K)
	    policy = v_I > v_N; % Chosen policy = 1 if investing
		end

		function [V,policy] = solve_last_interior(ix,jc,V,mp)
			p = mp.p;
			% Use grid index ix for x to find value of x
			x = mp.c(ix);
			% Use grid index jc for c to find value of c
			c = mp.c(jc);

			% Calculate expected value of value function V(s') = V(x',c')
			% conditional on investing a=1 and not investing a=0
			% EV_1 =  (1-prob(jc))*V(jc,jc) + prob(jc) * V(jc,jc) = V(jc,jc);
			v_I = mp.payoff(x) - mp.K(c) + mp.beta * V(jc,jc);
			v_N = mp.payoff(x)/(1-mp.beta); 

			V = max(v_I,v_N);
			policy = v_I>v_N;
		end

		function [V,policy] = solve_corner(ix,jc,V,mp)
	    p = mp.p;
	    % Use grid index ix for x to find value of x
	    x = mp.c(ix);
	    % at the corner c = x
	    c = x; % = mp.c(jc);
	    % Get markov probabilities
	    prob = mp.prob;
	    
	    v_N = ( mp.payoff(x) + mp.beta * (prob(jc)) * V(ix,jc+1) ) / ( 1 - mp.beta*(1-(prob(jc)))) ;
	    v_I = v_N - mp.K(c);

	    V = max(v_I,v_N);
	    policy = v_I>v_N;
		end

		function [V,policy] = solve_interior(ix,jc,V,mp)
	    p = mp.p;
	    % Use grid index ix for x to find value of x
	    x = mp.c(ix);
	    % Use grid index jc for c to find value of c
	    c = mp.c(jc);
	    % Get markov probabilities
	    prob = mp.prob;
	    
	    % Calculate expected value of value function V(s') = V(x',c')
	    % conditional on investing a=1: 
	    EV_1 =  (1-prob(jc))*V(jc,jc) + prob(jc) * V(jc,jc+1);
	    
	    v_I = mp.payoff(x) - mp.K(c) + mp.beta * EV_1;
	    v_N = ( mp.payoff(x) + mp.beta * prob(jc) * V(ix,jc+1) )/( 1 - mp.beta*(1-prob(jc)) );

	    V = max(v_I,v_N);
	    policy = v_I>v_N;   
		end

		function [V, P] = state_recursion(mp)
			% spp.state_recursion:  Solve social planners problmem with state recursion
			%
			%  [V, P] = spp.state_recursion(mp)
			%
			%  INPUT:
			%			mp:				Model parameters. See spp.setup
			%
			%  OUTPUT:
			%     V:        N x N matrix. Value function of social planner
			%			P:				Policy function of social planner
			% 							Column j of V and P corressponds to state of the art productions cost c = mp.c(j)
   		% 							Row i of V and P corressponds to productions cost x = mp.c(j)

			% number of possible states for state of the art production cost
			N = length(mp.c);

			% Initialize V and P to hold values of value function and policy function:
			V = NaN(N,N);
			P  = NaN(N,N);

			% Start by solving last corner:
			[V(N,N) , P(N,N)] = spp.solve_last_corner(mp);

			% Then solve interior of last layer:
			for ix = flip(1:N-1)
			    [V(ix,N) , P(ix,N)] = spp.solve_last_interior(ix,N,V,mp);
			end

			% Then solve corner and interior of the rest of the layers:
			for jc = flip(1:N-1)
			    
			     [V(jc,jc) , P(jc,jc)] = spp.solve_corner(jc,jc,V,mp);
			    
			    for ix = flip(1:jc-1)
			        
			        [V(ix,jc) , P(ix,jc)] = spp.solve_interior(ix,jc,V,mp);
			        
			    end
			end
		end

		function [V, P] = bellman(V, mp)
			% spp.bellman:  Bellman equation for solve social planner
			%
			%  [V, P] = spp.state_recursion(mp)
			%
			%  INPUT:
			%     V:        N x N matrix. Value function of social planner
			%			mp:				Model parameters. See spp.setup
			%
			%  OUTPUT:
			%     V:        N x N matrix. Value function of social planner
			%			P:				Policy function of social planner
			% 							Column j of V and P corressponds to state of the art productions cost c = mp.c(j)
   		% 							Row i of V and P corressponds to productions cost x = mp.c(j)

			% number of possible states for state of the art production cost
			N = length(mp.c);

			if V==0;
				V = NaN(N,N);
		    for j = 1:N
		       V(1:j,j) = 0;
		    end
		   	P = V;
		  end

			if size(V,1)~=N
				error('Number of columns in V must eqation number of gridpoints');
			elseif size(V,1)~=N
				error('Number of rows in V must eqation number of gridpoints');
			end

      for i=1:N % loop over current cost, x
      	x = mp.c(i);
        for j=i:N;	% loop ober state of the art cost, c
        	c = mp.c(j);
          v_N = mp.payoff(x) + mp.beta * (1-mp.prob(j)) * V(i,j);
          v_I = mp.payoff(x) - mp.K(c) + mp.beta * (1-mp.prob(j)) * V(j,j);
          if j < N
          	v_I = v_I + mp.beta * mp.prob(j) * V(j,j+1);
            v_N = v_N + mp.beta * mp.prob(j) * V(i,j+1);
         	end
		     	[V(i,j),P(i,j)] = max([v_N,v_I]);
		  	end % End loop over state of art mc
     	end % End loop over social planner mc
		end

		function [sim] = simulate(P, mp)
			% initial conditions
			ic(1)=1;
			ix(1)=1;
			u=rand(mp.T,1);
			for t=1:mp.T;
				I(t)=P(ix(t),ic(t)); 
				ix(t+1)=I(t)*ic(t)+(1-I(t))*ix(t);
				ic(t+1)=ic(t) + (u(t)<mp.prob(ic(t)));
				if ix(t)==mp.N;
					break;
					disp('got here')
				end
			end
			sim.ix=ix';
			sim.ic=ic';
			sim.I=I';			
			sim.t=(1:t+1)';
			sim.x=mp.c(sim.ix)'; 
			sim.c=mp.c(sim.ic)'; 
		end
	end % end of methods
end % end of classdef