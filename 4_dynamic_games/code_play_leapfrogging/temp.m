
  surf(c1,c2,p1');
  title(sprintf('P_1(c_1,c_2,%i)',stage));
  xlabel('c_1');
  ylabel('c_2');
  zlabel('P_1(c_1,c_2,0)');


  figure(2); % surface plot of firm 2's strategy
  surf(c1,c2,p2');
  title(sprintf('P_2(c_1,c_2,%i)',stage));
  xlabel('c_1');
  ylabel('c_2');
  zlabel('P_2(c_1,c_2,0)');

  figure(3); % surface plot of firm 1's value
  surf(c1,c2,v1p');
  title(sprintf('v_1(c_1,c_2,%i)=max[v_{N,1}(c_1,c_2,%i),v_{I,1}(c_1,c_2,%i))',stage,stage,stage));
  xlabel('c_1');
  ylabel('c_2');
  zlabel('v_1(c_1,c_2,0)');

  figure(4); % surface plot of firm 2's value
  surf(c1,c2,v2p');
  title(sprintf('v_2(c_1,c_2,%i)=max[v_{N,2}(c_1,c_2,%i),v_{I,2}(c_1,c_2,%i))',stage,stage,stage));
  xlabel('c_1');
  ylabel('c_2');
  zlabel('v_2(c_1,c_2,0)');
