classdef sgame
  	methods (Static)
		function p_a=br_a(p_b,x_a, alpha, beta) 
			%best response function, firm a
			p_a=1./(1+exp(-x_a*alpha+ p_b*x_a*(alpha-beta)));
		end

		function p_b=br_b(p_a,x_b, alpha, beta) 
			%best response function, firm a
			p_b=1./(1+exp(-x_b*alpha+ p_a*x_b*(alpha-beta)));
		end

		function p_a=br2_a(p_a,x_a,x_b, alpha, beta) 
			% second order best response function, firm a
			p_b=sgame.br_b(p_a,x_b, alpha, beta);
			p_a=sgame.br_a(p_b,x_a, alpha, beta); 
		end

	  	function p=FindStableEqb(p0, fx);
	  		% Procedure to find stable equilibrium, by successive approximations 
	  		% inputs 
	  		%		p0: staring point
	  		%   fx: second order best response
	  		% output
	  		% 	p: equilibrium 
	  		%   NOTE in case of multiple fixed points, p depends which p0 initializes the algorithm. 

	  		% Solve for fixed point using successive approximations on the second order best response function
			for i=1:100;
			  p=fx(p0);
			  if abs(p-p0)<1e-10
					tolerance=p-p0;
				 	% fprintf('Stable equilibrium found after %d iterations, tolerance = %1.4e\n', i,tolerance);
					break;
				end 
				p0=p;
			end
		end

	  	function p=FindUnstableEqb(l, u, fx);
	  		% Procedure to find unstable equilibrium using bisections between l and u 
	  		% inputs 
	  		%   l,u: lower and upper bounds on initial interval
	  		%   u: upper bound on initial interval
	  		%   fx: second order best response
	  		% output
	  		% 	p: unstable equilibrium 
	  		%   NOTE in case of multiple fixed points, p depends which p0 initializes the algorithm. 

	  		m=(l+u)/2; 
	  		fm=fx(m);
	  		fl=fx(l);
				fu=fx(u);
	    	convergence=0;         	
	    	p=nan(1,1);
				for i=1:200;
	        if fm>=m;
	            u=m;
	        elseif fm<m;
	            l=m;
	        elseif fm==m;
	            l=m;
	            u=m;
					end
	        m=(l+u)/2;
	    		fm=fx(m);
	         % fprintf('%d new [l,u] interval is [%g,%g]\n',i,l,u);
	        tolerance=abs(l-m);
	        if tolerance<1e-6
	        	p=m; 
	        	convergence=1;         	
					% fprintf('Unstable equilibrium found after %d iterations, tolerance = %1.4e\n', i,tolerance);
					break;
				end
			end
			if convergence==0
				fprintf('%d new [l,u] interval is [%g,%g]\n',i,l,u);
				warning('FindUnstableEqb did not converge')
			end
		end

  		function p=FindEqb(l, u, fx);
  			p0=sgame.FindStableEqb(0, fx);
			p1=sgame.FindStableEqb(1, fx);
			if abs(p0-p1)>1e-6 % more than one stable equilibrium found.
				% solve for Unstible Equilibrium using bisection algorithm
				pu=sgame.FindUnstableEqb(p0,p1, fx);
				p=[p0; pu; p1];
			else % unique equilibrium
				p=p0;	
			end
		end

		function [d_a,d_b, eqbinfo]=simdata_panel(x_a, x_b, alpha, beta, esr, T);
			M=numel(x_a); 
			if numel(esr)==1
				esr=repmat(esr, M,1);
			end
			for i=1:M 
				[d_a(:,i),d_b(:,i), eqbinfo(i)]=sgame.simdata(x_a(i), x_b(i), alpha, beta, esr(i),rand(T,2));
			end
		end

		function [p_a, p_b]=eqb(x_a, x_b, alpha, beta);
			% sgame.logl: computes all equilibria of static entry game
			% input: 
			% 	alpha, beta: parameters
			%   p_a, p_b: equilibrium probabilities
			%   k: selected equilibrium  

			% solve for ALL equilibria
			p_a=sgame.FindEqb(0, 1, @(p_a) sgame.br2_a(p_a,x_a,x_b, alpha, beta));
			p_b=sgame.br_b(p_a,x_b, alpha, beta);
		end

		function [logl, p_a, p_b, k]=logl(d_a,d_b, x_a, x_b, alpha, beta);
			global p_a_nfxp p_b_nfxp neqb_nfxp k;
			% sgame.logl: log likelihood function for NFXP estimation static entry game
			% input: 
			%		d_a, d_b:  Tx1 vectors of decision (market entry) indicator for firm a and b
			%		x_a, x_b:  Tx1 vectors with observed types for firm a and firm b
			% 	alpha, beta: parameters to be estimated
			% output: 
			% 	log likelihood
			%   p_a, p_b: equilibrium probabilities
			%   k: selected equilibrium  
			% solve for ALL equilibria
			p_a=sgame.FindEqb(0, 1, @(p_a) sgame.br2_a(p_a,x_a,x_b, alpha, beta));
			p_b=sgame.br_b(p_a,x_b, alpha, beta);

			neqb=numel(p_a);		% number of equilibria

			% compute log likelihood associated with each equilibrium
			logl=nan(neqb,1);	
			for ieqb=1:neqb
				logl_i=				d_a*log(p_a(ieqb)) + (1-d_a)*log(1-p_a(ieqb)) + ... 
											d_b*log(p_b(ieqb)) + (1-d_b)*log(1-p_b(ieqb));
				logl(ieqb)= sum(logl_i,1);
			end
			[logl, k]=max(logl); % take max over all equilibria
			neqb_nfxp=numel(p_a);
			p_a_nfxp=p_a(k);
			p_b_nfxp=p_b(k);
		end

		function [logl]=logl_panel(d_a,d_b, x_a, x_b, alpha, beta);
			global p_a_nfxp p_b_nfxp neqb_nfxp;
			M=numel(x_a); 
			p_a_nfxp=nan(M,1);
			p_b_nfxp=nan(M,1);
			neqb_nfxp=nan(M,1);
			logl=nan(M,1);
			for i=1:M 
				[logl(i), p_a, p_b, k]=sgame.logl(d_a(:,i),d_b(:,i), x_a(i), x_b(i), alpha, beta);
				neqb_nfxp(i)=numel(p_a);
				p_a_nfxp(i)=p_a(k);
				p_b_nfxp(i)=p_b(k);
			end
			logl=sum(logl);
		end

		function [d_a,d_b, eqbinfo]=simdata(x_a, x_b, alpha, beta, esr, randnum)
			p_a_all=sgame.FindEqb(0, 1, @(p_a) sgame.br2_a(p_a,x_a,x_b, alpha, beta));
			eqbinfo.neqb=numel(p_a_all);
			p_a=p_a_all(min(esr,eqbinfo.neqb));
			eqbinfo.esr=esr;
			p_b=sgame.br_b(p_a,x_b, alpha, beta);
			d_a=randnum(:,1)<p_a;
			d_b=randnum(:,2)<p_b;
			eqbinfo.p_a=p_a;
			eqbinfo.p_b=p_b;
		end

		function [logl]=logl_pml2step(d_a,d_b, x_a, x_b, alpha, beta, phat_a, phat_b);
			p_a=sgame.br_b(phat_b,x_a, alpha, beta);
			p_b=sgame.br_b(phat_a,x_b, alpha, beta);
			logl=				d_a*log(p_a) + (1-d_a)*log(1-p_a) + ... 
								  d_b*log(p_b) + (1-d_b)*log(1-p_b);
			logl= sum(logl,1);
		end

		function [theta1, phat1, fval]=npl_step(d_a,d_b, x_a, x_b, phat0, theta0);
				% pseudo likelihood
		      	logl=@(theta) sgame.logl_pml2step(d_a,d_b, x_a, x_b, theta(1), theta(2), phat0(1), phat0(2));
		      	options = optimset('Display','off');
		      	options.MaxIterations=10000;
		      	%  pseudo likelihood estimates, theta
				[theta1, fval, exitflag]=fminunc(@(theta)-logl(theta), theta0, options);
				if exitflag<=0
					theta1=nan(size(theta1));
				end
				% policy iteration using theta1
				phat1(1)=sgame.br_a(phat0(2),x_a, theta1(1), theta1(2)); % firm a
				phat1(2)=sgame.br_b(phat0(1),x_b, theta1(1), theta1(2)); % firm a
		end

    	function [logl]=logl_npl(d_a,d_b, x_a, x_b, alpha, beta);
			phat_a=mean(d_a);
			phat_b=mean(d_b);
			p_a=sgame.br_b(phat_b,x_a, alpha, beta);
			p_b=sgame.br_b(phat_a,x_b, alpha, beta);
			logl=				d_a*log(p_a) + (1-d_a)*log(1-p_a) + ... 
								  d_b*log(p_b) + (1-d_b)*log(1-p_b);
			logl= sum(logl,1);
		end

		function [logl]=logl_mpec(d_a,d_b, x_a, x_b, alpha, beta, p_a, p_b);
			% sgame.logl_mpec: log likelihood function for MPEC estimation static entry game
			% input: 
			%		d_a, d_b:  Tx1 vectors of decision (market entry) indicator for firm a and b
			%		x_a, x_b:  Tx1 vectors with observed types for firm a and firm b
			% 	alpha, beta, p_a, p_b: parameters to be estimated
			% output: 
			% 	log likelihood

			% compute log likelihood associated with each equilibrium
			logl=				d_a*log(p_a) + (1-d_a)*log(1-p_a) + ... 
									d_b*log(p_b) + (1-d_b)*log(1-p_b);
			logl=-sum(logl); % take max over all equilibria
		end

		function [c,ceq,DC,DCeq] = con_BNequations(x_a,x_b,alpha, beta, p_a, p_b)

			% Define and evaluate nonlinear equality constraints
			ceq=zeros(2,1);
			ceq(1)=p_a-sgame.br_a(p_b,x_a, alpha, beta);
			ceq(2)=p_b-sgame.br_b(p_a,x_b, alpha, beta);

			% ceq(1)=p_a-sgame.br_a(p_b,x_a, alpha, beta);
			% ceq(2)=p_b-sgame.br_b(p_a,x_b, alpha, beta); 


			% Define and evaluate nonlinear inequality constraints
			c = [];

			% Define and evaluate the constraint Jacobian (DC, DCeq).   
			if nargout >= 3
				error('Derivatives not implemented')
			DC= [];
			DCeq=[];
			end
		end % con_BNequations

  		function [logl]=logl_panel_mpec(d_a,d_b, x_a, x_b, theta);
			M=numel(x_a); 
			alpha=theta(1);
			beta=theta(2);
			p_a=theta(3:M+2);
			p_b=theta(M+3:end);
			for i=1:M 
				[logl(i)]=sgame.logl_mpec(d_a(:,i),d_b(:,i), x_a(i), x_b(i), alpha, beta, p_a(i), p_b(i));
			end
			logl=sum(logl);
		end
	
		function [c,ceq,DC,DCeq]=con_BNequations_panel(x_a, x_b, theta);
			M=numel(x_a); 
			alpha=theta(1);
			beta=theta(2);
			p_a=theta(3:M+2);
			p_b=theta(M+3:end);
			ceq=nan(2*M,1);
			for i=1:M 
				[c,ceq_i] = sgame.con_BNequations(x_a(i),x_b(i),alpha, beta, p_a(i), p_b(i));
				ceq(i)=ceq_i(1);
				ceq(i+M)=ceq_i(2);
			end
			DC=[]; DCeq=[];
		end
	end % end of methods
end % end of classdef