classdef lf
  methods (Static)
	  function [mp] = setup(mpopt)
	  	% set default parameters

			mp.Cmax= 5; % maximum state of the art cost
			mp.Cmin= 0; % minimum state of the art cost
			mp.nC = 4; % Number of states in technological proces (c1,c2,c3,,c_[mp.nC])
	  	mp.pf = 1;
			mp.k0 = 0;
			mp.k1 = 8.3;
	    mp.k2 = 1; 
	    mp.R = 0.05;
	    mp.Dt = 1;

			% update paramaters with input
	    if nargin>0
	        pfields=fieldnames(mpopt);
	        for i=1:numel(pfields);
	            mp.(pfields{i})=mpopt.(pfields{i});
	        end
	    end

			%% ---------- Set up parameters of the model -------
			
			% update dependent paramaters; 
			mp.p = [ones(1,mp.nC-1),0].*mp.pf; % transition probabilities for technological proces
	    mp.q = 1-mp.p;
			mp.C = linspace(mp.Cmax,mp.Cmin,mp.nC);

	    mp.beta = exp(-mp.R*mp.Dt);
	    mp.T = (mp.nC-1)*3 + 1;
	    mp.nESS = mp.nC * (mp.nC + 1) * (2*mp.nC+1)/6;

    	mp.firm1 = NaN(mp.nC,mp.nC,mp.nC);
	    mp.firm2 = mp.firm1;
    
    	for i = 1:mp.nC % Loop over technological levels
      	  mp.firm2(1:i,1:i,i) = repmat(mp.C(1:i),i,1);
        	mp.firm1(1:i,1:i,i) = mp.firm2(1:i,1:i,i)';   
    	end

    	mp.stage_index = lf.rlsIndex(mp.nC);
			% markovProbs

	    % functional forms
	    mp.K=   @(c) mp.k0+mp.k1/(1+mp.k2*c);				% Investment cost as function of state of the art production cost c
	    mp.payoff= @(x) (mp.p-x).*mp.dt;  	 	% flow payoffs per unit time ... 
	    mp.r1 = @(x1,x2) max(x2-x1,0)
    	mp.r2 = @(x1,x2) max(x1-x2,0)
    	mp.Phi = @(vN,vI) max([vN ; vI]);
		end

		function [ss,ESS] = state_recursion(ss, ESS, tau, mp)	
			% For each tau satisfying the condition the tau-specific
      % solver has to be applied.
      % For the Leapfrog model the solver varies according to whether:
      % Final layer: Corner, Edge, Interior or
      % Non-Final layer: Corner, Edge, Interior

			if tau == mp.T 
      	% Final layer corner
      	[ss,ESS] = lf.solve_last_corner(ss,ESS,mp);
        tau = tau - 1;
      end

      if tau == mp.T-1 
      	% % Final layer edge
	    	[ss,ESS] = lf.solve_last_edge(ss,ESS,mp);
	    	tau = tau - 1;
      end

      if tau == mp.T-2
      	% % Final layer interior
	      [ss,ESS] = lf.solve_last_interior(ss,ESS,mp);
	      tau = tau - 1;
	    end

      dothis = 1;
      while dothis==1 % infinite while loop... break on tau=0
        if mod(tau,3)==1
          % At the corner of a layer K
          ic = ceil((tau+2)/3);
          [ss,ESS] = lf.solve_corner(ss,ic,ESS,mp);
          tau = tau - 1;
          if tau == 0 % First layer is only corner so tau becomes 0 here
            break   % therefore break ... no more stages ...
          end
        end

	      if mod(tau,3)==0
					% At the edge of a layer
					ic = ceil((tau+2)/3);
					[ss, ESS] = lf.solve_edge(ss,ic,ESS,mp);
					tau = tau - 1;
	      end

				if mod(tau,3)==2
			    % At the interior of a layer
			    ic = ceil((tau+2)/3);
			    [ss, ESS] = lf.solve_interior(ss,ic,ESS,mp);
			    tau = tau - 1;
				end
      end % end of while tau>0
		end % end of state_recursion

		function [stage_index] = rlsIndex(nC)
      Ns = sum((1:nC).^2); % Number of state space points
      T = 1 + 3 * (nC-1);
      Ns_in_stages = ones(1,T);
      j = 1;
      l = nC;
      while l>1 % T, T-1,...,2  
        Ns_in_stages(1,j) = 1;
        j = j + 1;
        Ns_in_stages(1,j) = 2*(l-1);
        j = j + 1;
        Ns_in_stages(1,j) = (l-1).^2;
        j = j + 1;        
        l = l - 1;
      end
	  	stage_index = cumsum(Ns_in_stages);
		end

		function [tau] = cSS(N) 
			% INPUT: N is natural positive number = number of stages
			% OUTPUT: 1 x N struct tau representing stages of state space
			% PURPOSE: Create state space structure to hold info on identified
			% equilibria


			    % P1 player 1 probability invest
			    % vN1 value of not investing for player 1
			    % vI1 value of investing for player 1
			    % P2 player 2 probability of invest
			    % vN2 value of not investing for player 2
			    % vI2 value of investing for player 2 
			    
			    % Initialize datastructure for the state space
			    eq=struct('P1',[],'vN1',[],'vI1',[],'P2',[],'vN2',[],'vI2',[]);
			        for i = 1:N;
			            EQs(i,i,5).eq = eq; % 5 is because max 5 eqs ... consult litterature
			            tau(i).EQs = EQs;  % container for identified equilibriums
			            tau(i).nEQ = zeros(i,i); % container for number of eqs in (x1,x2,c) point
			        end
			   

			%  #  ##  ###  ####     State space with 4 stages.
			%     ##  ###  ####     4x4 Hashtag field is reached with complete
			%         ###  ####     technological development.
			%              ####     for each hashtag (x1,x2,c) - point in state space - max 5 eq's

		end

		function [index]=essindex(x,ic1,ic2,ic)
		  % INPUT: x is count of technological levels
		  % OUTPUT: ess index number for point (m,n,h) i state space
		  if all([ic1,ic2] == [ic,ic])
		      index = 1 + lf.div(x*(x+1)*(2*x+1),6) - lf.div(ic*(ic+1)*(2*ic+1),6);
		  elseif ic2 == ic
		      index = 1 + lf.div(x*(x+1)*(2*x+1),6) - lf.div(ic*(ic+1)*(2*ic+1),6) + ic1;
		  elseif ic1 == ic
		      index = 1 + lf.div(x*(x+1)*(2*x+1),6) - lf.div(ic*(ic+1)*(2*ic+1),6) + ic - 1 + ic2;
		  else
		      index = 1 + lf.div(x*(x+1)*(2*x+1),6) - lf.div(ic*(ic+1)*(2*ic+1),6) + 2*(ic - 1) + sub2ind([ic-1,ic-1],ic1,ic2);
		  end
		end

		function [out]=div(x,y)
    	out=floor(x./y);
		end

		function [pstar] = quad(a, b, c)
			% Solves:  ax^2  + bx + c = 0
			% but also always return 0 and 1 as candidates for probability of
			% investment
			d = b^2 - 4*a*c;
			if abs(a) < 1e-8
				pstar = [0. ; 1.;-c/b];
			else
				if d < 0
			    pstar = [0. ;1.];
				elseif d == 0.
					pstar = [0. ;1. ; -b/(2*a)];
				else
				 	pstar = [0. ;1. ; (-b - sqrt(d))/(2*a); (-b + sqrt(d))/(2*a)];
				end
			end
		end

		function [ess]=cESS(N)
		  % Create N x N x N array ess.index
		  % PURPOSE:
		  % ess.index(m,n,h) --> j
		  % where j is the index for ess.esr such that
		  % ess.esr(j)+1 is the equilibrium number played in state space
		  % point (m,n,h) this equilibrium is stored in the ss-object as
		  % ss(h).(m,n,j).eq = ss(h).(m,n,ess.esr(ess.index(m,n,h))+1)
		  ess.index=NaN(N,N,N);
		  for ic=1:N
		     for ic1=1:ic
		         for ic2=1:ic
		             ess.index(ic1,ic2,ic)  =  lf.essindex(N,ic1,ic2,ic)   ;        
		         end
		     end
		  end
		  % N*(N+1)*(2*N+1)/6 = sum(1^2 + 2^2 + 3^2 + ... + N^2)
		  ess.esr = zeros(1,N*(N+1)*(2*N+1)/6);
		  ess.bases = ones(1,N*(N+1)*(2*N+1)/6);
		  %ess.n = 1:(N*(N+1)*(2*N+1)/6);    
		end

		function [eq] = EQ(P1,vN1,vI1,P2,vN2,vI2)
			eq = struct('P1',P1,'vN1',vN1,'vI1',vI1,'P2',P2,'vN2',vN2,'vI2',vI2);
		end

		function [ss,ESS] = solve_last_corner(ss,ESS,mp)	    
	    % INPUT: global parameters cost and mp
	    % OUTPUT: Equilibrium of state space point (h,h,h) with h = mp.nC
	    
	    h = mp.nC; % Number of technological levels
	    c = mp.Cmin; % State of the art marginal cost for last tech. level
	    
	    % Both players have state of the art technology implemented ic1=ic2=c
	    
	    % If K>0 the vN1 = r1/(1-beta) .... geometric sum
	    vN1 = (mp.r1(c,c)+mp.beta * max(0,-mp.K(c)))  /  (1-mp.beta);
	    vI1 = vN1 - mp.K(c); 
	    P1 = vI1 > vN1;  % Equivalent to 0>mp.K(c);
	   
	    vN2 = (mp.r2(c,c)+mp.beta * max(0 , -mp.K(c)))  /  (1-mp.beta);
	    vI2 = vN2 - mp.K(c); 
	    P2 = vI2 > vN2; % Equivalent to 0>mp.K(c) and hence equal to P1;
	    
	    % OUTPUT is stored in ss
	    ss(h).EQs(h,h,1).eq = lf.EQ(P1,vN1,vI1,P2,vN2,vI2);
	    % Only one equilibrium is possible:
	    ss(h).nEQ(h,h) = 1;
	    ESS.bases(ESS.index(h,h,h)) = 1;
		end % end solve_last_corner

		function [ss,ESS] = solve_last_edge (ss,ESS,mp)
			% INPUT:
			% cost and mp are global parameters
			% ss state space structure (has info on eq's in corner of last layer)
			% OUTPUT:
			% Equilibria lf.EQ(P1, vN1, vI1, P2, vN2, vI2) for edge state space points
			% of the final layer:
			% Final layer <=> s=(x1,x2,c) with c = min(mp.C) 
			% Edge <=> s=(x1,x2,c) with x2 = c = min(mp.C) and x1 > c or
			% s=(x1,x2,c) with x1 = c = min(mp.C) and x2 > c

			ic = mp.nC; % Get the level of technology final layer
			c = mp.Cmin; % Get state of the art marginal cost for tech. of final layer

			h = 1; 
			% h is used to select equilibria in the corner of the final layer but there
			% is only ever 1 equilibria in the corner
			% If we did not apply this apriori knowledge we would have to use ESS
			% max(vN,vI | at the corner final layer)= mp.Phi(ss(ic).EQs(ic,ic,h).eq.vN1,ss(ic).EQs(ic,ic,h).eq.vI1)

			% Get the value of max choice in the corner of final layer s = (c,c,c)
			g1_ccc = max(ss(ic).EQs(ic,ic,h).eq.vN1,ss(ic).EQs(ic,ic,h).eq.vI1);
			g2_ccc = max(ss(ic).EQs(ic,ic,h).eq.vN2,ss(ic).EQs(ic,ic,h).eq.vI2);

			% Player 2 is at the edge s=(x1,x2,c) with x2=c=min(mp.C) and x1>c
			for ic1 = 1:ic-1
		    x1 = mp.C(ic1);
		    vI1 = mp.r1(x1,c) - mp.K(c)  + mp.beta * g1_ccc;
		    vN1search = @(z) mp.r1(x1,c) + mp.beta * mp.Phi(z,vI1) - z;
		    vN1 = fzero(vN1search,0);
		    P1 = vI1 > vN1;
		    
		    
		    vN2 = ( mp.r2(x1,c) + mp.beta * (P1*g2_ccc+(1-P1)*mp.Phi(0,-mp.K(c))) )  /  ( 1-mp.beta*(1-P1) );
		    vI2 = vN2 - mp.K(c);
		    P2 = vI2 > vN2;

		    ss(ic).EQs(ic1,ic,h).eq = lf.EQ(P1, vN1, vI1, P2, vN2 , vI2);
		    ss(ic).nEQ(ic1,ic) = 1;
		    ESS.bases(ESS.index(ic1,ic,ic)) = 1;
			end

			% Player 1 is at the edge s=(x1,x2,c) with x1=c=min(mp.C) and x2>c
			for ic2 = 1:ic-1
		    x2 = mp.C(ic2);
		    vI2 = mp.r2(c,x2) - mp.K(c) + mp.beta * g2_ccc;
		    vN2search = @(x) mp.r2(c, x2) + mp.beta*mp.Phi(x,vI2)-x;
		    vN2 = fzero(vN2search,0);
		    P2 = vI2 > vN2;
		    
		    
		    vN1 = (mp.r1(c, x2) + mp.beta*(P2*g1_ccc+(1-P2)*mp.Phi(0, -mp.K(c))))  /  ( 1-mp.beta*(1-P2) );
		    vI1 = vN1-mp.K(c);
		    P1 = vI1 > vN1;

		    ss(ic).EQs(ic, ic2, 1).eq = lf.EQ(P1, vN1, vI1, P2, vN2, vI2);
		    ss(ic).nEQ(ic, ic2) = 1;
		    ESS.bases(ESS.index(ic,ic2,ic)) = 1;
			end
		end % end solve_last_edge

		function [ss,ESS] = solve_last_interior(ss,ESS,mp)
			% outside loop
			ic = mp.nC;
			c = mp.C(ic);

		  g1 = @(iC1, iC2, iC) max(ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vN1,ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vI1);
		  g2 = @(iC1, iC2, iC) max(ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vN2,ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vI2);

      for ic1 = 1:ic-1 %Player 1 loop begin
     		for ic2 = 1:ic-1 %Player 2 loop begin                
	        % Player 1 -> leads to P2 candidates
	        a = mp.r1(mp.C(ic1), mp.C(ic2)) - mp.K(c) + mp.beta*g1(ic, ic2, ic); %check
	        b = mp.beta*(g1(ic, ic, ic)-g1(ic, ic2, ic)); % check
	        d = mp.r1(mp.C(ic1),mp.C(ic2));
	        e = mp.beta*g1(ic1, ic, ic);


	        b_0 = - mp.beta * b; % check 
	        b_1 = mp.beta * g1(ic1, ic, ic) + (mp.beta-1)*b - mp.beta*a; % check
	        b_2 = mp.r1(mp.C(ic1),mp.C(ic2)) + (mp.beta-1) * a; % check 


	        pstar2 = lf.quad(b_0, b_1, b_2); 
	        % always return 1 and 0 for the pure strategies


	        % Player 2 -> leads to P1 candidates
	        A = mp.r2(mp.C(ic1), mp.C(ic2)) - mp.K(c) + mp.beta*g2(ic1, ic, ic); 
	        B = mp.beta*(g2(ic, ic, ic)-g2(ic1, ic, ic));
	        D = mp.r2(mp.C(ic1),mp.C(ic2));
	        E = mp.beta*g2(ic, ic2, ic);

	        d_0 = - mp.beta * B;
	        d_1 = mp.beta*g2(ic, ic2, ic) + (mp.beta-1) * B - mp.beta*A;
	        d_2 = mp.r2(mp.C(ic1),mp.C(ic2)) + (mp.beta-1) * A;
	        
	        pstar1 = lf.quad(d_0, d_1, d_2);
	            
	         % Find equilibria based on candidates
	         % Number of equilibria found are 0 to begin with
	       	count = 0;
	        for i = 1:length(pstar1)
			      for j = 1:length(pstar2)
		          if all(ismember([i,j],[1,2])) % these are pure strategies
	              % If the polynomial is negative vI > vN
	              % hence player invests set exPj=1 else 0
	              % exP1 is best response to pstar2(j)
	              exP1 = b_2 + b_1 * pstar2(j) + b_0 * pstar2(j)^2 < 0 ;
	              exP2 = d_2 + d_1 * pstar1(i) + d_0 * pstar1(i)^2 < 0 ;

	              % check if both are playing best response
	              % in pure strategies. Players best response
	              % should be equal to the candidate to which
	              % the other player is best responding.
	              if abs(exP1 - pstar1(i)) < 1e-8 && abs(exP2-pstar2(j)) < 1e-8;
	                % if exP1=0 and pstar_i=0 true
	                % if exP1=1 and pstar_i=1 true
	                % Testing whether best response exP1 is
	                % equal to pstar1(i) to which Player 2
	                % is best responding ...
	                count = count + 1;
	                vI1 = a + b*pstar2(j); 
	                vN1 = (d + e*pstar2(j) + mp.beta*(1-pstar2(j))*(a+b*pstar2(j)))*pstar1(i)     +     (1-pstar1(i))*(d+e*pstar2(j))/(1-mp.beta*(1-pstar2(j)));
	                vI2 = A + B*pstar1(i); 
	                vN2 = (D + E*pstar1(i) + mp.beta*(1-pstar1(i))*(A+B*pstar1(i)))*pstar2(j)     +     (1-pstar2(j))*(D+E*pstar1(i))/(1-mp.beta*(1-pstar1(i)));

	                 ss(ic).EQs(ic1, ic2, count).eq = lf.EQ(pstar1(i),vN1,vI1,pstar2(j),vN2,vI2);
	              end
		          elseif i > 2 && j > 2 && pstar1(i) >= 0 && pstar2(j) >= 0 && pstar1(i) <= 1 && pstar2(j) <= 1
		              count = count + 1;
		              v1 = a + b * pstar2(j);
		              v2 = A + B * pstar1(i);
		              ss(ic).EQs(ic1, ic2, count).eq = lf.EQ(pstar1(i),v1,v1,pstar2(j),v2,v2);
		          end % end if
			      end % pstar2 loop
	        end % pstar1 loop
			    ss(ic).nEQ(ic1, ic2) = count; 
			    ESS.bases(ESS.index(ic1,ic2,ic)) = count;
        end %Player 2 loop end
      end %Player 1 loop end
		end % end solve_last_interior

		function [ss,ESS] = solve_corner(ss,ic,ESS,mp)
	    % ss(i).EQs(m,n,h).eq
	    % i is layer index
	    % m is level of technology of P1
	    % n is level of technology of P2
	    % h is selection equilibrium number given by esr
	    
	    % Corner: i given --> m=i and n=i
	    % We are solving the corner of not last stage ==> 
	    % (1) Technological development is possible ==> 
	    %       uncertainty over tech. development
	    % (2) Investment choice is inconsequetial ==>
	    %     No uncertainty about the other players investmentoutcome

	    % Get the marginal cost for the stage i under consideration
	    c = mp.C(ic);
	    
	    % Get probability of technological development
	    p = mp.p(ic);
	    
	    % Find 
	    % index for equilibrium selection h = 1 for simple selection rule
	    % Need ic+1 because ss(ic+1).EQs(ic,ic,h).eq is to be accessed
	    h = ESS.esr(ESS.index(ic,ic,ic+1))+1;
	    
	    vN1 = (  mp.r1(c,c)  +  mp.beta*p*max(ss(ic+1).EQs(ic,ic,h).eq.vN1,ss(ic+1).EQs(ic,ic,h).eq.vI1)   +   mp.beta*(1-p)*max(0,-mp.K(c)) )/(1-(1-p)*mp.beta);
	    vI1 = vN1 - mp.K(c);
	    vN2 = (  mp.r2(c,c)  +  mp.beta*p*max(ss(ic+1).EQs(ic,ic,h).eq.vN2,ss(ic+1).EQs(ic,ic,h).eq.vI2)   +   mp.beta*(1-p)*max(0,-mp.K(c)) )/(1-(1-p)*mp.beta);
	    vI2 = vN2 - mp.K(c);
	    
	    P1 = vI1 > vN1; % no investment uncertainty and no investments if K(c) > 0.
	    P2 = vI2 > vN2;

	    % Create output for return
	    ss(ic).EQs(ic,ic,1).eq = lf.EQ(P1, vN1, vI1, P2, vN2 , vI2);
	    ss(ic).nEQ(ic,ic) = 1;
	    ESS.bases(ESS.index(ic,ic,ic)) = 1;
	    % No update of ESS.bases is necessary in principle: "there can BE ONLY ONE
	    % equilibrium"  https://www.youtube.com/watch?v=sqcLjcSloXs
		end % end solve_corner

		function [ss, ESS] = solve_edge(ss,ic,ESS,mp)
			% INPUTS:
			% ss is state space
			% cost are global parameters see documentation
			% mp are global parameters see documentation
			% OUTPUT: returning state space with containing calculated solutions

			% PURPOSE: Function solves for quilibrium (P1, vN1, vI1, P2, vN2, vI2)
			% on the edges of the state space - though not corner - 

			% Get the marginal cost of layer ic
			c = mp.C(ic);

			% Get the probability of technological development occuring in layer ic
			p = mp.p(ic);

			% Creating some functions:
			% DEPENDENCIES: p as global (not passed as argument) and h
			% PURPOSE: Calculates expectations over technological development ...
			% technological development occurs: ss(iC+1)
			% technological development does not occur: ss(iC)

			H1 = @(iC1, iC2, iC) p*mp.Phi(ss(iC+1).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC+1))+1).eq.vN1,ss(iC+1).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC+1))+1).eq.vI1) ...
			+ (1-p)*mp.Phi(ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vN1,ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vI1);
			H2 = @(iC1, iC2, iC) p*mp.Phi(ss(iC+1).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC+1))+1).eq.vN2,ss(iC+1).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC+1))+1).eq.vI2) ...
			+(1-p)*mp.Phi(ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vN2,ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vI2);
			    
			% Efficiency ... why evaluate the call for each run of following loop? i is
			% constant in domain outside loop!! What changes in the function is the
			% ESS.

			% if at (x1,c,c) edge, with x1<c - Player 2 is at the edge.
			for ic1 = 1:ic-1 % index running over different technological levels for player1 not on edge
				% Get the marginal cost for player 1
				c1 = mp.C(ic1); 
				% First calculate vI1 depending only on known factors ... no uncertainty about Player 2 because he is at the edge
				vI1 = mp.r1(c1,c) - mp.K(c) + mp.beta*H1(ic,ic,ic);
				vN1search = @(z) mp.r1(c1,c) + mp.beta*(p*max(ss(ic+1).EQs(ic1,ic,ESS.esr(ESS.index(ic1,ic,ic+1))+1).eq.vN1, ss(ic+1).EQs(ic1, ic, ESS.esr(ESS.index(ic1,ic,ic+1))+1).eq.vI1)+(1-p)*max(z,vI1)) - z;
				vN1 = fzero(vN1search,0);
				P1 = vI1 > vN1; 

				vN2 = (mp.r2(c1,c) + mp.beta*(P1*H2(ic,ic,ic)+(1-P1)*(p*mp.Phi(ss(ic+1).EQs(ic1, ic,ESS.esr(ESS.index(ic1,ic,ic+1))+1).eq.vN2,ss(ic+1).EQs(ic1, ic,ESS.esr(ESS.index(ic1,ic,ic+1))+1).eq.vI2) + (1-p)*mp.Phi(0,-mp.K(c)))))/(1-mp.beta*(1-P1)*(1-p));
				vI2 = vN2 - mp.K(c);
				P2 = vI2 > vN2;

				ss(ic).EQs(ic1,ic,1).eq = lf.EQ(P1, vN1, vI1, P2, vN2, vI2);
				ss(ic).nEQ(ic1,ic) = 1;
				ESS.bases(ESS.index(ic1,ic,ic)) = 1;
			end % Exit player 1 not at edge loop

			% if at (c,x2,c) edge where x2<c - Player 1 is at the edge
			for ic2 = 1:ic-1
				c2 = mp.C(ic2);

				vI2 = mp.r2(c,c2) - mp.K(c) + mp.beta*H2(ic,ic,ic);
				vN2search = @(z) mp.r2(c,c2) + mp.beta*(p*max(ss(ic+1).EQs(ic, ic2,ESS.esr(ESS.index(ic,ic2,ic+1))+1).eq.vN2, ss(ic+1).EQs(ic, ic2, ESS.esr(ESS.index(ic,ic2,ic+1))+1).eq.vI2) + (1-p)*max(z,vI2)) - z;
				vN2 = fzero(vN2search,0);
				P2 = vI2 > vN2;

				vN1 = (mp.r1(c,c2) + mp.beta*(P2*H1(ic,ic,ic)+(1-P2)*(p*mp.Phi(ss(ic+1).EQs(ic, ic2, ESS.esr(ESS.index(ic,ic2,ic+1))+1).eq.vN1,ss(ic+1).EQs(ic, ic2, ESS.esr(ESS.index(ic,ic2,ic+1))+1).eq.vI1)+(1-p)*mp.Phi(0,-mp.K(c)))))/(1-mp.beta*(1-P2)*(1-p));
				vI1 = vN1-mp.K(c);
				P1 = vI1 > vN1;

				ss(ic).EQs(ic,ic2,1).eq = lf.EQ(P1, vN1, vI1, P2, vN2, vI2);
				ss(ic).nEQ(ic,ic2) = 1;
				ESS.bases(ESS.index(ic,ic2,ic)) = 1;
				% No update of ESS.bases is necessary: "there can BE ONLY ONE
				% equilibrium"  https://www.youtube.com/watch?v=sqcLjcSloXs
			end % Exit player 2 not at edge loop
		end % end solve_edge

		function [ss , ESS]= solve_interior(ss,ic,ESS,mp)
			% INPUT 
			% ss is state space structure with solutions for final layer edge and
			% corner
			% ic is the level of technology for which to solve
			% ESS is struc with information holding ESS.esr being equilibrium selection
			% rule and ESS.bases being the bases of the ESS.esr's

			c = mp.C(ic);
			for ic1 = 1:ic-1;
		    for ic2 = 1:ic-1;
		 			[ss , ESS] = lf.find_interior(ss,mp,ic1,ic2,ic,c,ESS);
		    end             
			end
		end

		function [ss , ESS] = find_interior(ss,mp,ic1,ic2,ic,c,ESS)
			% INPUT
			% ss is state space
			% mp and cost holds global parameters
			% ic1 and ic2 are indexes for player 1 and player 2 such that mp.C(ic1)
			% and mp.C(ic2) are marginal cost of the players
			% k is the level of technology
			% c=mp.C(k) state of the art marginal cost ... could be found inside
			% function
			% 

			% get probability of technological development
			p = mp.p(ic);
			q = 1-p;

			% h is used for selected equilibrium in state realized when technology
			% develops hence ic+1 in ESS.index(ic1,ic2,ic+1)
			h = ESS.esr(ESS.index(ic1,ic2,ic+1))+1;



			% H1(iC1, iC2, iC) = (1-pi)*?(?[iC+1].EQs[iC1, iC2, ESS[iC1,iC2,iC+1]].vN1,?[iC+1].EQs[iC1, iC2, ESS[iC1,iC2,iC+1]].vI1) + pi*?(?[iC].EQs[iC1, iC2, ESS[iC1,iC2,iC]].vN1,?[iC].EQs[iC1, iC2, ESS[iC1,iC2,iC]].vI1)
			% H2(iC1, iC2, iC) = (1-pi)*?(?[iC+1].EQs[iC1, iC2, ESS[iC1,iC2,iC+1]].vN2,?[iC+1].EQs[iC1, iC2, ESS[iC1,iC2,iC+1]].vI2) + pi*?(?[iC].EQs[iC1, iC2, ESS[iC1,iC2,iC]].vN2,?[iC].EQs[iC1, iC2, ESS[iC1,iC2,iC]].vI2)


			H1 = @(iC1, iC2, iC) p*mp.Phi(  ss(iC+1).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC+1))+1).eq.vN1 , ss(iC+1).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC+1))+1).eq.vI1  ) ...
			+ (1-p)*mp.Phi(  ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vN1 , ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vI1  );
			H2 = @(iC1, iC2, iC) p*mp.Phi(  ss(iC+1).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC+1))+1).eq.vN2 , ss(iC+1).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC+1))+1).eq.vI2  ) ...
			+ (1-p) * mp.Phi(  ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vN2 , ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vI2  );

			a = mp.r1( mp.C(ic1),mp.C(ic2) ) - mp.K(c) + mp.beta * H1(ic,ic2,ic); %check
			b = mp.beta * (   H1(ic,ic,ic) - H1(ic,ic2,ic)  ); %check
			d = mp.r1( mp.C(ic1),mp.C(ic2) )   + mp.beta * p * mp.Phi( ss(ic+1).EQs(ic1,ic2,h).eq.vN1 , ss(ic+1).EQs(ic1,ic2,h).eq.vI1  ) ;
			e = mp.beta * H1(ic1,ic,ic)         - mp.beta * p * mp.Phi( ss(ic+1).EQs(ic1,ic2,h).eq.vN1 , ss(ic+1).EQs(ic1,ic2,h).eq.vI1  ) ; 

			pa = - mp.beta * (1-p) * b;
			pb = e + ( mp.beta * (1-p) -1) * b - mp.beta * (1-p) * a; 
			pc = d + ( mp.beta * (1-p) -1 ) * a;

			% Solve for p2 mixed strategy ... but also returns 1 and 0 for pure
			pstar2 = lf.quad(pa,pb,pc);

			A = mp.r2(mp.C(ic1),mp.C(ic2)) - mp.K(c) + mp.beta * H2(ic1,ic,ic);
			B = mp.beta * ( H2(ic,ic,ic) - H2(ic1,ic,ic) );
			D = mp.r2(mp.C(ic1),mp.C(ic2)) + mp.beta * p * mp.Phi( ss(ic+1).EQs(ic1,ic2,h).eq.vN2 , ss(ic+1).EQs(ic1,ic2,h).eq.vI2 );
			E = mp.beta * H2(ic,ic2,ic)       - mp.beta * p * mp.Phi( ss(ic+1).EQs(ic1,ic2,h).eq.vN2 , ss(ic+1).EQs(ic1,ic2,h).eq.vI2 );

			qa = - mp.beta * (1-p) * B;
			qb = E + ( mp.beta * (1-p) - 1 ) * B - mp.beta * (1-p) * A;
			qc = D + ( mp.beta * (1-p) - 1 ) * A;

			pstar1 = lf.quad(qa, qb, qc);

    
			count = 0;
			for i = 1:length(pstar1);
				for j = 1:length(pstar2);
					 if all(ismember([i,j],[1,2]));
				     exP1 = pc + pb * pstar2(j) + pa * pstar2(j)^2 < 0 ;
				     exP2 = qc + qb * pstar1(i) + qa * pstar1(i)^2 < 0 ;

				    if abs(exP1 - pstar1(i)) < 1e-7 && abs(exP2-pstar2(j)) < 1e-7;
			        count = count + 1;
			        vI1 = a + b*pstar2(j);
			        vN1 = (d + e*pstar2(j) + mp.beta*q*(1-pstar2(j))*(a+b*pstar2(j)))*pstar1(i)+(1-pstar1(i))*(d+e*pstar2(j))/(1-mp.beta*q*(1-pstar2(j)));
			        vI2 = A + B*pstar1(i);
			        vN2 = (D + E*pstar1(i) + mp.beta*q*(1-pstar1(i))*(A+B*pstar1(i)))*pstar2(j)+(1-pstar2(j))*(D+E*pstar1(i))/(1-mp.beta*q*(1-pstar1(i)));

			        ss(ic).EQs(ic1, ic2, count).eq = lf.EQ(pstar1(i),vN1,vI1,pstar2(j),vN2,vI2);
				    end
				    
				    elseif i > 2 && j > 2 && pstar1(i) >= 0 && pstar2(j) >= 0 && pstar1(i) <= 1 && pstar2(j) <= 1
			        count = count + 1;
			        v1 = a + b * pstar2(j);
			        v2 = A + B * pstar1(i);
			        ss(ic).EQs(ic1, ic2, count).eq = lf.EQ(pstar1(i),v1,v1,pstar2(j),v2,v2);
					 end
				end % end j loop
			end % end i loop
    
	    ss(ic).nEQ(ic1,ic2) = count;
	    ESS.bases(ESS.index(ic1,ic2,ic)) = count;
		end

		function [Vu,Vx,Vz] = vscatter(V,jitter,sigma,adjust,VM)
			% INPUT: N x 2+ matrix with value function values in first two columns
			% jitter is indicator if 1 jitter is added
			% sigma is std. of jitter noise
			% adjust is indicator if 1 adjustment using weights is used
			% VM is used to draw triangle ... should be social planners value for
			% similar model as the one used to find V
			% a is size increase to smallest all points (use to increase small bubbles)
			% d is factor to decrease max(weights) because some equilibria are just
			% very large in number
	    V = V(:,1:2);
	    a = 5;
	    d = 0.005;
	    [Vu,Vx,Vz] = unique(round(V+0.000001,3),'rows');
	    N = length(Vu(:,1));
	    
	   	if adjust==1
	    	weights = ones(N,1);
	      for i=1:N
					weights(i,1) = sum( Vz == i); 
	      end
	      weights = a+weights./(d*max(weights));
	  	else
	    	weights = ones(length(Vu),1);
	    end
	    
    	if jitter==1
      	e1=normrnd(0,sigma,N,1);
        e2=normrnd(0,sigma,N,1);
    	else
      	e1 = zeros(N,1);
       	e2 = e1;
    	end
    
    	scatter(Vu(:,1)+e1,Vu(:,2)+e2,weights,'filled')
    	grid on
    
    	if nargin>4
       line([0,VM],[VM,0]) 
       line([0,0],[0,VM])
       line([0,VM],[0,0])
    	end
		end
	end % end of methods
end % end of classdef