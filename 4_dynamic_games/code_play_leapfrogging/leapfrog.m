% leapfrog.m: play the leapfrogging game, with or without a human opponent at the keyboard
%
%             Fedor Ishakov, John Rust, Bertel Schjerning, June, 2017

close all;

setup;

robot_player=1;  

deterministic_onestep=0; % deterministic one step technology

autoplay=0;  % set to 1 to have two computer players play the game,
             % set to 0 to play against player 1, the robot player, whose strategy is selected above

doplotattheend=1; % plot results of the game at the end
doplotsaswego=0;  % plot the game history as game evolves

shade_profits=1; % enter 1 to shade regions where one of the firms is the low cost firm
                 % indicating their profits while in the low cost position with red (for firm 1)  
                 % and blue (for firm 2). 

deadsteps=30;    % if there is no action for more than this number, invoke early termination of game

voice=1;  % 0 to turn voice off on mac

print_hist=1; %print game history in the end


% do not change any code below unless you know what you are doing!

if (robot_player ~= 1);
    [ir,v]=state_recursion; % solve for the monopoly investment strategy using state recursion
end;

continue_playing=1;
ax=[]; %place for plot axes handle

if (~autoplay);
 fprintf('Welcome to the leapfrogging game!\nYou are firm 2, your cost is c2.\nFirm 1 cost is c1 and state of the art cost is c\n\n');
 if ismac && voice
  !say Welcome to the leapfrogging game
 end
end;

while (continue_playing);

fprintf('\n\n');

stage=nstates;   % initialize time step through the game. Game will terminate
                 % when stage reaches 0 or if there are too many periods of
                 % no action in the game
c=c0;
c1=c0;
c2=c0;
t=0;
laststage=0;
gameover=0;
noaction=0;
earlytermination=0;
i1=0;
i2=0;
cumprof1=0;
cumprof2=0;
discprof1=0;
discprof2=0;
lastp2=0;
gamehist=[];

while (stage > 0);

   t=t+1;

   kv=kf(c);

   if (stage == 1 & c1 <= 0 & c2 <= 0);
      gameover=1;
   else;
      if (noaction > deadsteps);
        gameover=1;
        earlytermination=1;
      end;
   end;

   p=max(c1,c2);

   % df=bet^(t-1);  % update discount factor to calculate discounted profits
   df=bet^t;  % discount to the start of the game (period 0)

   currentprofit1=0;
   currentprofit2=0;
   if (c1 < c2);
       % discprof1=bet*discprof1+df*(p-c1);
       currentprofit1=p-c1;
       discprof1=discprof1+df*(p-c1); %discount to the game start
       cumprof1=cumprof1+(p-c1);
   else;
       % discprof1=bet*discprof1;
   end;    
   if (c2 < c1);
       % discprof2=bet*discprof2+df*(p-c2);
       currentprofit2=p-c2;
       discprof2=discprof2+df*(p-c2);
       cumprof2=cumprof2+(p-c2);
   else;
       % discprof2=bet*discprof2;
   end;    

   if (robot_player == 1);

    [p1 p2 v1 v2]=stagegame_eq(c1,c2,stage);

    if (~autoplay);
     truep2=p2;
     if (t> 1 &gamehist(t-1,6));
        lastp2=0;
     end;

     fprintf('\n%24s\n',sprintf('Time period t=%d',t));
     fprintf('%s\n',repmat('-',1,44));
     fprintf('%24s     %10.4f\n','State of the art cost',c);
     fprintf('%24s     %10.4f\n','Investment cost',kv);
     fprintf('%24s     %10.4f\n','Price',max(c1,c2));
     fprintf('%s\n',repmat('-',1,44));
     fprintf('%24s%10s%10s\n','','Firm 1','You');
     fprintf('%s\n',repmat('-',1,44));
     fprintf('%24s%10.4f%10.4f\n','Production cost',c1,c2);
     fprintf('%24s%10.4f%10.4f\n','Current profits',currentprofit1,currentprofit2);
     fprintf('%24s%10.4f%10.4f\n','Cumulative profits',cumprof1,cumprof2);
     fprintf('%24s%10.4f%10.4f\n','Discounted profits @t=0',discprof1,discprof2);
     fprintf('%s\n',repmat('-',1,44));
     % fprintf('t=%i (c1,c2,c)=(%g,%g,%g) price: %g firm 1 cumulative profits %g  your cumulative profits %g\n',...
     %   t,c1,c2,c,max(c1,c2),cumprof1,cumprof2);
     reply=input(sprintf('Enter your probability of investing or hit return to use this value: %g ',lastp2),'s');
     if isempty(reply);
         p2=lastp2;
     else;
         p2=str2double(reply);
         lastp2=p2;
     end;
    end;

    if (p1 <= 0 & p2 <= 0);

     if (autoplay); 
       fprintf('t=%i stage=%i no investment equilibrium by either firm (c1,c2,c)=(%g,%g,%g)\n',t,stage,c1,c2,c);
     end;
     i1=0;
     i2=0;
     if (stage == 1);
        gameover=1;
     end;

    else;

     if (p1 > 0 & p2 <= 0);

      if (p1 >= 1); 

        if (autoplay);
        fprintf('t=%i stage=%i p1=%g p2=0  firm 1 invests (pure strategy) (c1,c2,c)=(%g,%g,%g)\n',t,stage,p1,c1,c2,c);
        end;
        i1=1;
        i2=0;

      else;

        u=rand(1,1);

        if (u <= p1);

         if (autoplay);
         fprintf('t=%i stage=%i p1=%g p2=0  firm 1 invests (mixed strategy) (c1,c2,c)=(%g,%g,%g)\n',t,stage,p1,c1,c2,c);
         end;
         i1=1;
         i2=0;

        else;

         if (autoplay);
         fprintf('t=%i stage=%i p1=%g p2=0  firm 1 does not invest (mixed strategy) (c1,c2,c)=(%g,%g,%g)\n',t,stage,p1,c1,c2,c);
         end;
         i1=0;
         i2=0;

        end;

      end;

     elseif (p1<=0 & p2 > 0);

      if (p2 >= 1); 

        if (autoplay);
        fprintf('t=%i stage=%i p1=0 p2=%g  firm 2 invests (pure strategy) (c1,c2,c)=(%g,%g,%g)\n',t,stage,p2,c1,c2,c);
        end;
        i2=1;
        i1=0;

      else;

        u=rand(1,1);

        if (u <= p2);

         if (autoplay); 
         fprintf('t=%i stage=%i p1=0 p2=%g  firm 2 invests (mixed strategy) (c1,c2,c)=(%g,%g,%g)\n',t,stage,p2,c1,c2,c);
         end;
         i2=1;
         i1=0;

        else;

         if (autoplay);
         fprintf('t=%i stage=%i p1=0 p2=%g  firm 2 does not invest (mixed strategy) (c1,c2,c)=(%g,%g,%g)\n',t,stage,p2,c1,c2,c);
         end;
         i1=0;
         i2=0;

        end;

      end;
 
     else;

       i1=0;
       i2=0;

       u=rand(1,1);
       if (u <= p1);
        i1=1;
       end;

       u=rand(1,1);
       if (u <= p2);
        i2=1;
       end;

       if (autoplay);
         if (~i1 & ~i2);
          fprintf('t=%i stage=%i p1=%g p2=%g  (mixed strategy eq, neither firm invests) (c1,c2,c)=(%g,%g,%g)\n',t,stage,p1,p2,c1,c2,c);
         elseif (~i1 & i2);
          fprintf('t=%i stage=%i p1=%g p2=%g  (mixed strategy eq, firm 1 no invest, firm 2 invests) (c1,c2,c)=(%g,%g,%g)\n',...
           t,stage,p1,p2,c1,c2,c);
         elseif (i1 & ~i2);
          fprintf('t=%i stage=%i p1=%g p2=%g  (mixed strategy eq, firm 1 invests, firm 2 no invest) (c1,c2,c)=(%g,%g,%g)\n',...
           t,stage,p1,p2,c1,c2,c);
         else;
          fprintf('t=%i stage=%i p1=%g p2=%g  (mixed strategy eq, both firms invest) (c1,c2,c)=(%g,%g,%g)\n',...
           t,stage,p1,p2,c1,c2,c);
         end;
       end;


     end; % end of if branch for p1 and p2

     %truep2=p2;
     if (t> 1 &gamehist(t-1,6));
        lastp2=0;
     end;

    end;


   else;  % this branch plays the monopoly equilibrium
          % if robot_player = 0 then then firm 1 never invests and firm 2 is the monopolist
          % if robot_player = 2 then firm 1 is the monopolist and firm 2 never invests

    if (autoplay);

     if (robot_player == 0);
         i1=0;
         i2=ir(trindx(c2,c));
         p1=0;
         p2=i2;
         fprintf('t=%i stage=%i monopoly equilibrium firm 1 never invests p2=%g (c1,c2,c)=(%g,%g,%g)\n',...
             t,stage,p2,c1,c2,c);
     else;
         i2=0;
         i1=ir(trindx(c1,c));
         p2=0;
         p1=i1;
         fprintf('t=%i stage=%i monopoly equilibrium firm 2 never invests p1=%g (c1,c2,c)=(%g,%g,%g)\n',...
             t,stage,p1,c1,c2,c);
     end;

    else;
     
     if (t> 1 &gamehist(t-1,6));
        lastp2=0;
     end;

     fprintf('\n%24s\n',sprintf('Time period t=%d',t));
     fprintf('%s\n',repmat('-',1,44));
     fprintf('%24s     %10.4f\n','State of the art cost',c);
     fprintf('%24s     %10.4f\n','Investment cost',kv);
     fprintf('%24s     %10.4f\n','Price',max(c1,c2));
     fprintf('%s\n',repmat('-',1,44));
     fprintf('%24s%10s%10s\n','','Firm 1','You');
     fprintf('%s\n',repmat('-',1,44));
     fprintf('%24s%10.4f%10.4f\n','Production cost',c1,c2);
     fprintf('%24s%10.4f%10.4f\n','Current profits',currentprofit1,currentprofit2);
     fprintf('%24s%10.4f%10.4f\n','Cumulative profits',cumprof1,cumprof2);
     fprintf('%24s%10.4f%10.4f\n','Discounted profits @ t=0',discprof1,discprof2);
     fprintf('%s\n',repmat('-',1,44));
     % fprintf('t=%i (c1,c2,c)=(%g,%g,%g) price: %g firm 1 cumulative profits %g  your cumulative profits %g\n',...
     %   t,c1,c2,c,max(c1,c2),cumprof1,cumprof2);

     reply=input(sprintf('   enter your probability of investing or hit return to use this value: %g ',lastp2),'s');
     if isempty(reply);
         p2=lastp2;
     else;
         p2=str2double(reply);
         lastp2=p2;
     end;

     u=rand(1,1);
     if (u <= p2);
        i2=1;
     else;
        i2=0;
     end;

     if (robot_player == 0);
       i1=0;
       p1=0;
       truep2=ir(trindx(c2,c));
     else;
       i1=ir(trindx(c1,c));
       p1=i1;
       truep2=0;
     end;

    end;

   end;



  %kv=kf(c);

  if (i1);
     discprof1=discprof1-df*kv;
     cumprof1=cumprof1-kv;
  end;
  if (i2);
     discprof2=discprof2-df*kv;
     cumprof2=cumprof2-kv;
  end;

  % update gamehist array

  if (autoplay);
    gamehist=[gamehist; [t stage p1 p2 i1 i2 c1 c2 c cumprof1 cumprof2 discprof1 discprof2]];
  else;
    gamehist=[gamehist; [t stage p1 p2 i1 i2 c1 c2 c cumprof1 cumprof2 discprof1 discprof2 truep2]];
  end;

   if doplotsaswego && ~isempty(gamehist)
      if isempty(ax)
        % position=get(0,'ScreenSize');
        % position([1 2])=0;
        % fig=figure('Color',[1 1 1],'Position',position);
        fig=figure('Color',[1 1 1]);
        ax=axes('Parent',fig);
      else
        cla(ax,'reset');
      end
      plotgame(gamehist,autoplay,shade_profits,cumprof1,cumprof2,ax);
   end

   if (gameover);
      if (earlytermination);
         fprintf('\nNo action for %i periods: terminating game\n',noaction-1);
      else;
         fprintf('\nEnd game edge state reached: game over\n');
      end;
      break;
   end;

  % update endogenous states to reflect investments, if any

  if (i1);
     c1=c;
  end;

  if (i2);
     c2=c;
  end;

  % now draw next period state of the art cost, c (the exogenous state)

  if (deterministic_onestep);

     newstage=stage-1;

  else;

   if (laststage ~= stage);

     cumprob=zeros(stage,1);
     cumprob(1)=stp(1,stage);
     for i=2:stage;
       cumprob(i)=cumprob(i-1)+stp(i,stage);
     end;

   end;

   u=rand(1,1);
   newstage=min(find(u < cumprob));

  end;

  if (newstage <= 0);
    break;
  end;

  if (newstage < stage);
     fprintf('\n *** Technological innovation c=%g -> c=%g\n     Investment cost to acquire this new technology: %g\n',...
     c,cgrid(newstage),kv);
     if ismac && voice && ~deterministic_onestep
      !say Technological innovation
     end
     c=cgrid(newstage);
     stage=newstage;
     noaction=0;
  else;

     if (~i1 & ~i2);

       noaction=noaction+1;

     end;
 
  end;

  laststage=stage;

end;

if (autoplay);
  fprintf('cumulative profits firm 1: %g  cumulative profits firm 2: %g\n',cumprof1,cumprof2);
else;
  fprintf('\nGame over!\n')
  fprintf('%s\n',repmat('-',1,44));
  fprintf('%24s%10s%10s\n','','Firm 1','You');
  fprintf('%s\n',repmat('-',1,44));
  fprintf('%24s%10.4f%10.4f\n','Cumulative profits',cumprof1,cumprof2);
  fprintf('%24s%10.4f%10.4f\n','Discounted profits',discprof1,discprof2);
  fprintf('%s\n',repmat('-',1,44));
  if cumprof1>cumprof2
    fprintf('Firm 1 wins!\n')
     if ismac && voice
      !say Game over. You lost.
     end
  else
    fprintf('Congratulations, you are the winner!\n');
     if ismac && voice
      !say Game over. You won!
     end
   end
  % fprintf('Cumulative profits firm 1: %g\nYour cumulative profits (as firm 2): %g\n',cumprof1,cumprof2);

  if print_hist
    fprintf('Below is the game history, your probability of investing is recorded in column 4, the equilibrium (mixed strategy) probability is the last column\n');
    fprintf('    t        stage          p1        p2        i1        i2    c1        c2        c       cumprof1  cumprof2   discprof1 discprof2  true p2\n');
    gamehist
  end
  
end;

if doplotattheend && doplotsaswego
    cla(ax,'reset');
    plotgame(gamehist,autoplay,shade_profits,cumprof1,cumprof2,ax);      
elseif doplotattheend
    plotgame(gamehist,autoplay,shade_profits,cumprof1,cumprof2);
end;

     fprintf('\n\n');
     reply=input(sprintf('\nPlay another game? (0 or hit return to quit, 1 to play again)'),'s');
     if isempty(reply);
         continue_playing=0;
     else;
         continue_playing=str2double(reply);
         if (isnan(continue_playing));
            continue_playing=0;
         end;
     end;
  

end;

