% matlab program to solve endgame equilibria

global k bet;

c1=5.1;  % by convention make firm 1 the high cost firm
c2=.26316;
bet=.95;
k=5;
c2=k*(1-bet)/bet;
%k=k+.0001;
c1t=c1;
c2t=c2;
%c1=c2t;
%c2=c1t;

% check for no investment equilibrium

 if (c1 > c2);

  iv_profit1=-k+bet*c2/(1-bet);
  iv_profit1_gain=iv_profit1;

  iv_profit2=-k+c1-c2+bet*c1/(1-bet);
  ni_profit2=c1-c2+bet*(iv_profit2);

  iv_profit2_gain=iv_profit2-ni_profit2;

 else;

  iv_profit2=-k+bet*c1/(1-bet);
  iv_profit2_gain=iv_profit2;

  iv_profit1=-k+c2-c1+bet*c2/(1-bet);
  ni_profit1=c2-c1+bet*(iv_profit1);

  iv_profit1_gain=iv_profit1-ni_profit1;

 end;

  fprintf('profit for firm 1 (c1=%g) if it invests but firm 2 does not: %g\n',c1,iv_profit1);
  fprintf('profit for firm 2 (c2=%g) if it invests but firm 1 does not: %g\n',c2,iv_profit2);
  fprintf('gain in profit from firm 2 from investing relative to not investing (assuming firm 1 does not invest): %g\n',iv_profit2_gain);

  if (iv_profit1 < 0 & iv_profit2_gain < 0);

    fprintf('No investment MPE\n');

  elseif (iv_profit1 < 0 & iv_profit2_gain > 0);

    fprintf('Pure strategy MPE where firm 1 (high cost firm) does not invest  and firm 2 (low cost firm)  invests\n');

  else;

    fprintf('Mixed strategy equilibrium where both firms invest\n');

    if (c1 > c2);

    p2=1-k*(1-bet)/(bet*c2);
    fprintf('Firm 2 investment probability: %g  firm 1 payoff: 0\n',p2);

    % solve quadratic equation for p1

    a=-bet*bet*c1/(1-bet);
    b=bet*c1/(1-bet)-bet*(c1-c2-k);
    c=-k;

    p1u=(-b+sqrt(b^2-4*a*c))/(2*a);
    p1l=(-b-sqrt(b^2-4*a*c))/(2*a);
    p1=1-p1u;
    v2=(c1-c2)/(1-bet*p1u);
    v2invest=c1-c2-k+bet*p1u*c1/(1-bet);
     
    fprintf('Firm 1 investment probability: %g  firm 2 payoff in mixed MPE: %g payoff from investing for sure %g\n',1-p1u,v2,v2invest);
    if (v2invest > v2)
      fprintf('value of investing for sure is higher than value under mixed strategy equilibrium\n');
    end;
   
    if (1-p1l > 0 & 1-p1l < 1)
      fprintf('2nd mixed strategy solution for firm 1: %g\n',1-p1l);
    end;

    else;

    p1=1-k*(1-bet)/(bet*c1);
    fprintf('Firm 1 investment probability: %g  firm 2 payoff: 0\n',p1);

    % solve quadratic equation for p2

    a=-bet*bet*c2/(1-bet);
    b=bet*c2/(1-bet)-bet*(c2-c1-k);
    c=-k;

    p2u=(-b+sqrt(b^2-4*a*c))/(2*a);
    p2l=(-b-sqrt(b^2-4*a*c))/(2*a);
    p2=1-p2u;
    v1=(c2-c1)/(1-bet*p2u);
    v1invest=c2-c1-k+bet*p2u*c2/(1-bet);
     
    fprintf('Firm 2 investment probability: %g  firm 1 payoff in mixed MPE: %g payoff from investing for sure %g\n',p2,v1,v1invest);
    if (v1invest > v1)
      fprintf('value of investing for sure is higher than value under mixed strategy equilibrium\n');
    end;
   
    if (1-p2l > 0 & 1-p2l < 1)
      fprintf('2nd mixed strategy solution for firm 2: %g\n',1-p2l);
    end;

    end;

  end;


  x=(0:.01:1)';
  sx=size(x,1);
  vn=zeros(sx,1);
  vi=zeros(sx,1);

  if (c1 > c2);

  for i=1:sx;
    vn(i)=(c1-c2)/(1-bet*(1-x(i)));
    vi(i)=c1-c2-k+bet*(1-x(i))*c1/(1-bet);
  end;

  figure(1);
  hold on;
  plot(x,vn,'b-','Linewidth',2);
  plot(x,vi,'r-','Linewidth',2);
  title('Values of investing and not-investing for firm 2, endgame');
  xlabel('Probability firm 1 invests');
  ylabel('Firm 2 values');
  yl=ylim;
  line([p1 p1],yl,'Color','k');
  text(p1+.01,yl(1)*.5+.4*yl(2),sprintf('p_1=%4.2f',p1));
  legend('Value of not investing','Value of investing','Location','Northeast');
  axis('square');
  hold off;

  else;

  for i=1:sx;
    vn(i)=(c2-c1)/(1-bet*(1-x(i)));
    vi(i)=c2-c1-k+bet*(1-x(i))*c1/(1-bet);
  end;

  figure(1);
  hold on;
  plot(x,vn,'b-','Linewidth',2);
  plot(x,vi,'r-','Linewidth',2);
  title('Values of investing and not-investing for firm 1, endgame');
  xlabel('Probability firm 2 invests');
  ylabel('Firm 1 values');
  yl=ylim;
  line([p2 p2],yl,'Color','k');
  text(p2+.01,yl(1)*.5+.4*yl(2),sprintf('p_1=%4.2f',p2));
  legend('Value of not investing','Value of investing','Location','Northeast');
  axis('square');
  hold off;

  end;
  figure(2);  % surface plot of firm 1's strategy; 

  c1=(0:.01:.3)';
  c2=(0:.01:.3)';
  sc2=size(c2,1);
  sc1=size(c1,1);
  p1=zeros(sc1,sc2);
  p2=zeros(sc1,sc2);
  v1p=zeros(sc1,sc2);
  v2p=zeros(sc1,sc2);

  for i=1:sc1;
   for j=1:sc2;
      [p1(i,j),p2(i,j),v1p(i,j),v2p(i,j)]=endgame_eq(c1(i),c2(j));
      if (p1(i,j) < 0);
        fprintf('c1=%g c2=%g p1=%g\n',c1(i),c2(j),p1(i,j));
      end;
   end;
  end;

  surf(c1,c2,p1');
  title('P_1(c_1,c_2,0)');
  xlabel('c_1');
  ylabel('c_2');
  zlabel('P_1(c_1,c_2,0)');


  figure(3); % surface plot of firm 2's strategy
  surf(c1,c2,p2');
  title('P_2(c_1,c_2,0)');
  xlabel('c_1');
  ylabel('c_2');
  zlabel('P_2(c_1,c_2,0)');

  figure(4); % surface plot of firm 1's value
  surf(c1,c2,v1p');
  title('v_1(c_1,c_2,0)=max[v_{N,1}(c_1,c_2,0),v_{I,1}(c_1,c_2,0))');
  xlabel('c_1');
  ylabel('c_2');
  zlabel('v_1(c_1,c_2,0)');

  figure(5); % surface plot of firm 2's value
  surf(c1,c2,v2p');
  title('v_2(c_1,c_2,0)=max[v_{N,2}(c_1,c_2,0),v_{I,2}(c_1,c_2,0))');
  xlabel('c_1');
  ylabel('c_2');
  zlabel('v_2(c_1,c_2,0)');
