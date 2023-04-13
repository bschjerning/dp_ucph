% trindx.m: program to compute index for a triangular array of points
%           (c1,c) where c1 is the marginal cost of production of a monpolist
%           producer and c is the marginal cost under the state of the art.
%           Both c1 and c are discretized to assume ngp possible values on the
%           vector cgrid between cmin and cmax, which are specified in setup.m
%           There is the natural constaint that c1 >= c, i.e. the firm's legacy
%           technology can never have a marginal cost lower than the state of the
%           art that steadily improves, albeit stochastically. The firm can adopt
%           the state of the art technology by paying a fixed cost K(c) to invest
%           in a new production facility with the state of the art production technology.
%
%           For purposes of crrying out the DP solution by policy iteration
%           we "vectorize" the value function V(c1,c) as an n x 1 vector where
%           n=ngp*(ngp+1)/2 following the ordering 
%
%           1       c1(1)   c(1)
%           2       c1(2)   c(2)
%           3       c1(2)   c(1)
%           4       c1(3)   c(3)
%           5       c1(3)   c(2)
%           6       c1(3)   c(1)
%           7        ...    ...
%           n-ngp   c1(ngp) c(n)
%           n-ngp+1 c1(ngp) c(n-1)
%           ...       ...    ...
%           n-1     c1(ngp) c(2)
%           n       c1(ngp) c(1)
%
%           If we denote (c1,c0 by (i,j) (c1=cgrid(i) and c=cgrid(j)  with i >= j)
%           then we have  index=i*(i-1)/2+i-j+1 
%
%           John Rust, Georgetown University, July 2012

  function  index=trindx(c1,c)

  global cgrid ngp;

  c1i=find(c1 == cgrid);

  if (size(c1i) == [0 1]);

    fprintf('Error trindx: c1 argument not found in cgrid, c1=%g. Argument must be restricted to the cgrid.\n',c1);
    index=nan;
    return;

  end;

  ci=find(c == cgrid);

  if (size(ci) == [0 1]);

    fprintf('Error trindx: c argument not found in cgrid, c=%g. Argument must be restricted to the cgrid.\n',c);
    index=nan;
    return;

  end;

  if (c1 < c);
  
     fprintf('Error trindx: c1 is less than c, violating state space constraint (c1,c)=(%g,%g)\n',c1,c);
     index=nan;
     return;

  end;

  index=(c1i*(c1i-1)/2)+c1i-ci+1;
