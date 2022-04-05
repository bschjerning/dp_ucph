classdef rls
  methods (Static)
		function [ESS, TAU, out] = solve(G,ss,ESS0, stage_index)

			rlsp.maxEQ  =  50000; % maximum number of iterations
			rlsp.print  = 500;    % print every rlsp.print equilibria (0: no print, 1: print every, 2: print every second)

			% initialize matrices
			TAU = NaN(rlsp.maxEQ,1);
			tau = numel(stage_index); % start RLS at last stage
			iEQ = 1;
			ESS(iEQ)=ESS0;
			while iEQ <= rlsp.maxEQ
        TAU(iEQ,1) = tau;
        [ss,ESS(iEQ)] = G(ss,ESS(iEQ), tau);    
        
        if (mod(iEQ,rlsp.print)==0) 
        	fprintf('ESR(%d).esr  : [', iEQ);
        	fprintf(' %d', ESS(iEQ).esr);
        	fprintf(']\n');
        	fprintf('ESR(%d).bases: [', iEQ);
        	fprintf(' %d', ESS(iEQ).bases);
        	fprintf(']\n\n');
        end
               	
       	if nargout>2
       		out(iEQ)=rls.output(ss, ESS(iEQ));
       	end

        ESS(iEQ+1) = rls.addOne(ESS(iEQ));
        changeindex = min(find((ESS(iEQ+1).esr-ESS(iEQ).esr)~=0));
        tau = sum(changeindex<=stage_index)-1; % tau0 is found
        
        if all(ESS(iEQ+1).esr==-1)
            break
        end
        
        iEQ = iEQ + 1;
			end % End of recursive lexicographical search
			TAU=TAU(1:iEQ);
		end

		function [out]=output(ss, ESS)
				out.MPEesr = ESS.esr;
				out.V1 = max([ss(1).EQs(1,1,1).eq.vN1,ss(1).EQs(1,1,1).eq.vI1]);
				out.V2 = max([ss(1).EQs(1,1,1).eq.vN2,ss(1).EQs(1,1,1).eq.vI2]);
		end 

		function [ESS] = addOne(ESS)
	    %if x[1,1,1] == -1
	    %    throw(error("This ESS has already overflown!"))
	    %end
	    n = length(ESS.esr) ;   
	    X = zeros(1,n);
	    R = 1;
	    for i = flip(1:n)
				X(i) = mod(ESS.esr(i) + R,ESS.bases(i)); 
				% mod(a,b) does division and returns the remainder given as a-div(a,b)*b
				R = lf.div(ESS.esr(i) + R,ESS.bases(i)); 
				% div(a,b) does division and truncates - rounding down - to nearest integer  .... floor(a/b)
	    end

	    if R > 0
				% When exiting the loop R > 0 occurs when all ESS.number is max allowed
				% which is 1 below the base.
				% println("No more equilibria to check.")
				ESS.esr = -1*ones(1,n);
	    else
				ESS.esr = X;
	    end
		end

	end % end of methods
end % end of classdef