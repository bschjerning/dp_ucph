% stp.m: state transition probability function
%        based on a triangular probability specification
%        John Rust, Georgetown University, January 2017


   function [stpv]=stp(nl,n);

     % n is an integer index for the current cost state
     % nl is an integer index for the new state 
     % by definition cgrid is the grid of cost states, equally spaced from [0,c0]
     % so c, c' must be elements of the cost grid with c' <= c with probability 1
     % cgrid(1)=0, the lowest possible state of the art cost
     % cgrid(nstates)=c0 the initial and highest possible state of the art cost
     

     global nstates c0 cgrid dtp;

     if (dtp);  % deterministic transitiion probability

       if (nl == n-1);
        stpv=1;
       else;
        stpv=0;
       end;
  
    else;

     rissp=.2;

     riss=1-cgrid(n)*rissp/(1+c0);  % probability that the state of the art cost does not drop 
                       % to a lower level next period

     a=.2; % a parameter defining the triangular distribution for cost improvements:
           % must be in the unit interval  

     if (nl > n);
       fprintf('new index for cost state, nl,  (%i) must be less or equal to the index of the current cost state, %i\n',nl,n);
     end;

     if (nl <= 0);
       fprintf('index for new cost state must be an integer great or equal to 1. The value you entered is %i\n',nl);
     end;

     if (n == 1);

       stpv=1;

     else;

        b=(1-riss)*a/(n-1);

        if (n == 2);
         if (nl == n);
             stpv=riss;
         else;
             stpv=1-riss;
         end;
        else;
          if (n == nl);
            stpv=riss;
          else;
            s=2*nstates*(1-a)*(1-riss)/(c0*(n-1)*(n-2));
            stpv=b+s*(nl-1)*c0/nstates;
          end;
        end;

     end;

   end;
