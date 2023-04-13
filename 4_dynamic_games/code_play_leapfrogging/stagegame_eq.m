% stagegame_eq: function to compute the mixed strategy equilibrium  of a given
%               stage game of the leapfrogging game, recursively.
%               Stages are defined by the global cgrid vector and the nstates
%               integer global specifying the number of possible values of the
%               state of the art production cost, equally spaced over the interval
%               [0,c0]. By convention, stage=1 denotes the end game.
%               For speed, this program assumes c1 < c2 and uses symmetry of the solution
%               to compute results when c1 >= c2. Results are stored in vmat, and if not
%               found there, then are computed via a recursive call to this function.
% 
%               John Rust, Georgetown University, January 2017

 function [p1,p2,v1,v2]=stagegame_eq(c1,c2,stage);

 global k k1 k2 bet c0 cgrid nstates vmat;

 % look to see if the equilibrium has already been computed and stored in vmat,
 % if so return the already computed values, otherwise compute them an add them to vmat

 % fprintf('calling stagegame_eq with c1=%g c2=%g stage=%i\n',c1,c2,stage);

 if (size(vmat,1) > 0);
     
   i=find(vmat(:,1)==c1 & vmat(:,2) == c2 & vmat(:,3)==stage);

   if (size(i,1) > 0);

     p1=vmat(i,6);
     p2=vmat(i,7);
     v1=vmat(i,4);
     v2=vmat(i,5);
     %fprintf('using previously computed equilibrium at (c1,c2,stage)=(%g,%g,%i)\n',c1,c2,stage);
     if (size(p1,1) > 1);
       fprintf('error p1 has more than 1 elements\n');
       i
       p1
     end;
     return;

   end;

 end;

 % the code below assumes c1 <= c2.  If c1 > c2 we use symmetry of the solution

 symmetry=0;
 if (c1 > c2);
    symmetry=1;
    c1t=c1;
    c2t=c2;
    c1=c2t;
    c2=c1t;
 end;

 minc=min(c1,c2);
 c=cgrid(stage);
 k=kf(c);

 if (stage == 1);

  iv_profit2=-k+bet*c1/(1-bet);
  if (iv_profit2 > 0);
     ni_profit2=bet*iv_profit2;
  else;
     ni_profit2=0;
  end;
  iv_profit2_gain=iv_profit2-ni_profit2;

  iv_profit1=-k+c2-c1+bet*c2/(1-bet);
  if (iv_profit1 > 0);
   ni_profit1=c2-c1+bet*(iv_profit1);
   if (ni_profit1 > iv_profit1);
      ni_profit1=(c2-c1)/(1-bet); 
   end;
  else;
   ni_profit1=(c2-c1)/(1-bet);
  end;

  iv_profit1_gain=iv_profit1-ni_profit1;

 else;

  sp=stp(stage,stage);

  h1cc1c=h1(c,c1,stage);
  h1c1c2c=h1(c1,c2,stage);
  %h1c2c1c=h1(c2,c1,stage);
  h1cc2c=h1(c,c2,stage);

  boundary_val=(c2-c+bet*h1cc2c)/(1-bet*sp);
  boundary_val1=(c1-c+bet*h1cc1c)/(1-bet*sp);

  iv_profit2=-k+bet*(h1cc1c+sp*boundary_val1);
  %ni_profit2=bet*h1c2c1c/(1-bet*sp);
  ni_profit2=0;
  %if (ni_profit2 < iv_profit2);
  if (0 < iv_profit2);
    ni_profit2=bet*sp*iv_profit2; 
    %ni_profit2=bet*h1c2c1c+bet*sp*iv_profit2; 
  end;
  iv_profit2_gain=iv_profit2-ni_profit2;

  iv_profit1=-k+c2-c1+bet*(h1cc2c+sp*boundary_val);
  ni_profit1=(c2-c1+bet*h1c1c2c)/(1-bet*sp);
  if (ni_profit1 < iv_profit1);
    ni_profit1=c2-c1+bet*h1c1c2c+bet*sp*iv_profit1;
  end;
  iv_profit1_gain=iv_profit1-ni_profit1;

if (iv_profit2_gain >= 0 & iv_profit1_gain < 0);
fprintf('stage=%i c1=%g c2=%g iv_propfit1_gain=%g iv_profit2_gain=%g\n',stage,c1,c2,iv_profit1_gain,iv_profit2_gain);
end;
%diff=h1cc2c-h1c1c2c-h1cc1c;
%if (diff < 0 )
%fprintf('stage=%i c1=%g c2=%g iv_propfit1_gain=%g iv_profit2_gain=%g diff = %g\n',stage,c1,c2,iv_profit1_gain,iv_profit2_gain,h1cc2c-h1c1c2c-h1cc1c);
%end;
 
 end;

 %fprintf('stage %i (c1,c2,c)=(%g,%g,%g)\n',stage,c1,c2,c);
 %fprintf('stage %i gain to investing for firm 1: %g\n',stage,iv_profit1_gain);
 %fprintf('stage %i gain to investing for firm 2: %g\n',stage,iv_profit2_gain);
 %fprintf('bet=%g k=%g\n',bet,k);
 %fprintf('profit for firm 1 (c1=%g) if it invests but firm 2 does not: %g\n',c1,iv_profit1);
 %fprintf('gain in profit from firm 2 from investing relative to not investing (assuming firm 1 does not invest): %g\n',iv_profit2_gain);
  
 %fprintf('bet=%g k=%g\n',bet,k);
 %fprintf('profit for firm 1 (c1=%g) if it invests but firm 2 does not: %g\n',c1,iv_profit1);
 %fprintf('gain in profit from firm 2 from investing relative to not investing (assuming firm 1 does not invest): %g\n',iv_profit2_gain);
  
 if (iv_profit1_gain < 0 & iv_profit2_gain < 0);

    p1=0;
    p2=0;
    v1=ni_profit1;
    v2=0;

 elseif (iv_profit1_gain < 0 & iv_profit2_gain > 0);

    %fprintf('Pure strategy MPE where firm 1 does not invest and firm 2 invests, stage %i (c1,c2)=(%g,%g) v1I-v1N=%g v2I-v2N=%g firm 1 boundary val %g\n',stage,c1,c2,iv_profit1_gain,iv_profit2_gain,boundary_val);
    if (c2 > c1);
       fprintf('WARNING 1: Type 4 equilibrium found here\n');
    end;
        
    p1=0;
    p2=1;
    v1=ni_profit1;
    v2=iv_profit2;

 elseif (iv_profit1_gain > 0 & iv_profit2_gain < 0);

    %fprintf('Pure strategy MPE where firm 1 invests and firm 2 does not invest, stage %i (c1,c2)=(%g,%g) v1I-v1N=%g v2I-v2N=%g\n',stage,c1,c2,iv_profit1_gain,iv_profit2_gain);
    if (c1 > c2);
       fprintf('WARNING 2: Type 4 equilibrium found here\n');
    end;
    p1=1;
    p2=0;
    v1=iv_profit1;
    v2=ni_profit2;

 else;

  if (stage == 1);

    p1=1-k*(1-bet)/(bet*c1);

    % solve quadratic equation for p2

    a=-bet*bet*c2/(1-bet);
    b=bet*c2/(1-bet)-bet*(c2-c1-k);
    c=-k;

    p2u=(-b+sqrt(b^2-4*a*c))/(2*a);
    p2l=(-b-sqrt(b^2-4*a*c))/(2*a);
    p2=1-p2u;
    v1=(c2-c1)/(1-bet*p2u);
    v1invest=c2-c1-k+bet*p2u*c2/(1-bet);
    v2=0;

  else;

    p1=1-k/(bet*(h1cc1c+sp*boundary_val1));

    if (c1 == c2);

       p2=p1;
       v1=0;
       v1invest=0;
       v2=0;

    else;

%  fprintf('h1cc2c=%g boundary_val=%g  H1(c,c2,c)=H1(%g,%g,%g)=%g p2t=%g\n',h1cc2c,boundary_val,c,c2,c,h1cc2c+sp*boundary_val,p2t);
%  fprintf('h1cc1c=%g boundary_val1=%g  H2(c1,c,c)=H2(%g,%g,%g)=%g p1=%g\n',h1cc1c,boundary_val1,c1,c,c,h1cc1c+sp*boundary_val1,p1);

     if (sp <= 0);

       p2=1-k/(bet*(h1cc2c-h1c1c2c));
       v1=c2-c1+bet*(1-p2)*h1c1c2c;

     else;

       a=-bet*bet*sp*(h1cc2c+sp*boundary_val);
       b=bet*(h1cc2c+sp*boundary_val-sp*(c2-c1-k));
       c=-k-bet*h1c1c2c;

       p2u=(-b+sqrt(b^2-4*a*c))/(2*a);
       p2l=(-b-sqrt(b^2-4*a*c))/(2*a);
       p2=1-p2u;
       v1=(c2-c1+bet*h1c1c2c)/(1-bet*sp*p2u);
       v1invest=c2-c1-k+bet*p2u*(h1cc2c+sp*boundary_val);

       if (v1invest > v1+1e-10)
        fprintf('Warning: value of investing for sure is higher than value under mixed strategy equilibrium\n');
       end;
   
       if (1-p2l > 0 & 1-p2l < 1)
        fprintf('2nd mixed strategy solution for firm 2: %g\n',1-p2l);
       end;

     end;

     v2=0;
 
   end;

  end;

 end;

 if (symmetry);

   v1t=v1;
   v2t=v2;
   p1t=p1;
   p2t=p2;
 
   v1=v2t;
   v2=v1t;
   p1=p2t;
   p2=p1t;

   c1=c1t;
   c2=c2t;

   if (v1 > 0);
     fprintf('Warning: high cost firm has positive value (c1,c2,stage)=(%g,%g,%i) (v1,v2)=(%g,%g)\n',c1,c2,stage,v1,v2);
   end;

 else;
   
   if (v2 > 0);
     fprintf('Warning: high cost firm has positive value (c1,c2,stage)=(%g,%g,%i) (v1,v2)=(%g,%g)\n',c1,c2,stage,v1,v2);
   end;
  
 end;

 if (size(p1,1) > 1);
    fprintf('error stagegame_eq: size p1 > 1\n');
    p1
 end;

 vmat=[vmat; [c1 c2 stage v1 v2 p1 p2]];
 if (p1 ~= p1);
  vmat=[vmat; [c2 c1 stage v2 v1 p2 p1]];
 end;

