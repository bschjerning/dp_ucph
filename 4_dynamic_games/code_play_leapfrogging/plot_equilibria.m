% matlab program to plot stage game equilibria for the mixed strategy MPE of the leapfrogging game
%        John Rust, Georgetown University, January 2017

setup;

  global vmat;
  vmat=[];
  stage=18;
  c=cgrid(stage);

  figure(1);  % surface plot of firm 1's strategy; 

%  c0=2;

  c1=(c:.05:c0)';
  c2=(c:.05:c0)';
  sc2=size(c2,1);
  sc1=size(c1,1);
  p1=zeros(sc1,sc2);
  p2=zeros(sc1,sc2);
  v1p=zeros(sc1,sc2);
  v2p=zeros(sc1,sc2);

  for i=1:sc1;
   for j=1:sc2;
      [p1(i,j),p2(i,j),v1p(i,j),v2p(i,j)]=stagegame_eq(c1(i),c2(j),stage);
      if (p1(i,j) < 0);
        fprintf('c1=%g c2=%g p1=%g\n',c1(i),c2(j),p1(i,j));
      end;
      if (c1(i) == c2(j) & (v1p(i,j) > 0 | v2p(i,j) > 0));
         fprintf('problem: positive diagonal values at c1=%g c2=%g stage=%i v1=%g v2=%g\n',c1(i),c2(j),stage,v1p(i,j),v2p(i,j));
      end;
   end;
  end;

  surf(c1,c2,p1');
  title(sprintf('P_1(c_1,c_2,%g)',cgrid(stage)));
  xlabel('c_1');
  ylabel('c_2');
  zlabel(sprintf('P_1(c_1,c_2,%g)',cgrid(stage)));


  figure(2); % surface plot of firm 2's strategy
  surf(c1,c2,p2');
  title(sprintf('P_2(c_1,c_2,%g)',cgrid(stage)));
  xlabel('c_1');
  ylabel('c_2');
  zlabel(sprintf('P_2(c_1,c_2,%g)',cgrid(stage)));

  figure(3); % surface plot of firm 1's value
  surf(c1,c2,v1p');
  title(sprintf('v_1(c_1,c_2,%g)=max[v_{N,1}(c_1,c_2,%g),v_{I,1}(c_1,c_2,%g))',cgrid(stage),cgrid(stage),cgrid(stage)));
  xlabel('c_1');
  ylabel('c_2');
  zlabel(sprintf('v_1(c_1,c_2,%g)',cgrid(stage)));

  figure(4); % surface plot of firm 2's value
  surf(c1,c2,v2p');
  title(sprintf('v_2(c_1,c_2,%g)=max[v_{N,2}(c_1,c_2,%g),v_{I,2}(c_1,c_2,%g))',cgrid(stage),cgrid(stage),cgrid(stage)));
  xlabel('c_1');
  ylabel('c_2');
  zlabel(sprintf('v_2(c_1,c_2,%g)',cgrid(stage)));
