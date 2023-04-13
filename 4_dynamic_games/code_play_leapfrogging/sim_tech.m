% simulate technological evolution
%
%             Fedor Ishakov, John Rust, Bertel Schjerning, June, 2017

close all;

setup;

deterministic_onestep=0; % deterministic one step technology
deadsteps=100;    % if there is no action for more than this number, invoke early termination of game


% plot
position=get(0,'ScreenSize');
position([1 2])=0;
h=figure('Color',[1 1 1],'Position',position);
hold on;

sims=10;

for sim=1:sims;  % run many simulations
  hist =[];

  stage=nstates;
  stepcount = nstates;
  
  while (stepcount > 0);

    stepcount = stepcount-1;

    hist(end+1)=cgrid(stage);

    % fprintf("%d %1.4f \n",stage,cgrid(end))

    % now draw next period state of the art cost, c (the exogenous state)
    if (deterministic_onestep);

      newstage=stage-1;

    else

      cumprob=zeros(stage,1);
      cumprob(1)=stp(1,stage);
      for i=2:stage;
        cumprob(i)=cumprob(i-1)+stp(i,stage);
      end;

      u=rand(1,1);
      newstage=min(find(u < cumprob));
      
      if stage ~=newstage
        if newstage==1
          stepcount = 20;
        else          
          stepcount = deadsteps;
      end
    end


    end;

    stage=newstage;
    

  end;

  stairs(max(hist-(sim-sims/2)*0.025,0),'LineWidth',2);
  set(gca,'FontSize',16);
  pause(.25);
  % keyboard

end
hold off;


