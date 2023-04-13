% state_recursion.m: solves the dynamic cost minimization via the state-recursion DP algorithm
%                    John Rust, Georgetown University, April 2017

function [ir,v]=state_recursion;

   global nstates cgrid bet dt;   % dimension of points in the state space

   c=cgrid(1);

   n=nstates*(nstates+1)/2;
   v=zeros(n,1);
   ir=zeros(n,1);
   cvec=zeros(n,1);
   c1vec=zeros(n,1);

   for i=1:nstates;

      c1=cgrid(i);

      v0=dt*c1/(1-bet);
      v1=kf(c)+dt*c1;
      ind=trindxi(i,1);

      if (v0 < v1);
       
       ir(ind)=0;
       v(ind)=v0;

      else;

       ir(ind)=1;
       v(ind)=v1;

      end;

      cvec(ind)=c;
      c1vec(ind)=c1;
     
   end;  

   for j=2:nstates;

     c=cgrid(j);
     pnti=stp(j,j);     % probability of no technological innovation this period
     den=(1-bet*pnti);

     %fprintf('\ncalculating v(c1,c) at discrete state j=%i c=%g\n',j,c);

     for i=j:nstates;

       c1=cgrid(i);
       tmp=0;
       for k=1:j-1;
         tmp=tmp+stp(k,j)*v(trindxi(i,k));
       end;
       ind=trindxi(i,j);
       if (i == j);

          v(ind)=(dt*c+bet*tmp)/den;
          ir(ind)=0;
          %fprintf('edge solution v(c1(%i),c(%i))=%g, ind=%i c1=%g c=%g ir=0\n',i,j,v(ind),ind,c1,c); 

       else;

          tmp1=0;
          for k=1:j-1;
              tmp1=tmp1+stp(k,j)*v(trindxi(j,k));
          end;

          v0=(dt*c1+bet*tmp)/den;
          v1=dt*c1+kf(c)+bet*tmp1+bet*pnti*v(trindxi(j,j));

          if (v0 < v1);
 
            ir(ind)=0;
            v(ind)=v0;
   %fprintf('gh ir(ind)=%i ind=%i v0=%g v1=%g\n',ir(ind),ind,v0,v1);

          else;

            ir(ind)=1;
   %fprintf('gh ir(ind)=%i ind=%i\n',ir(ind),ind);
            v(ind)=v1;

          end;

          %fprintf('calculating v(c1(%i),c(%i))=%g, index=%i c1=%g c=%g v0=%g v1=%g ir=%i\n',i,j,v(ind),ind,c1,c,v0,v1,ir(ind)); 
       end;
     
     end; 
 
   end;
