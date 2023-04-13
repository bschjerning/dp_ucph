function plotgame(gamehist,autoplay,shade_profits,cumprof1,cumprof2,ax);

  if isempty(gamehist)
    return;
  end

  if ~exist('ax')
    position=get(0,'ScreenSize');
    position([1 2])=0;
    fig=figure('Color',[1 1 1],'Position',position);
    ax=axes('Parent',fig);
  end
  hold(ax,'all');
  set(ax,'Ylim',[max(0,floor(min(gamehist(:,9))-1)),max(gamehist(:,9))+.5]);
  set(ax,'Xlim',[min(gamehist(:,1)),max(10,max(gamehist(:,1))+1)]);

   T=size(gamehist,1);

   x=(1:T+1)';
   c=gamehist(:,9);
   c=[c; c(T)];
   c1=gamehist(:,7);
   c1=[c1; c1(T)];
   c2=gamehist(:,8);
   c2=[c2; c2(T)];

   xp_c=[];
   cp_c=[];
   nbreaks_c=0;
   lastcp=0;
   lastbreakt=0;
   lasti=0;
   vlinex_c=cell(0);
   vliney_c=cell(0);
   hlinex_c=cell(0);
   hliney_c=cell(0);
   for i=1:T+1;
      if (i > 1);
         if (c(i-1)>c(i));
              nbreaks_c=nbreaks_c+1;
              if (nbreaks_c == 1);
                cp_c=[c(1:i-1); NaN];
                xp_c=[x(1:i-1); NaN];
              else;
                if (lastbreakt == i-1);
                  cp_c=[cp_c(1:lastcp); c(lastbreakt); NaN];
                  xp_c=[xp_c(1:lastcp); x(lastbreakt); NaN];
                else;
                  if (lasti == i-1);
                    cp_c=[cp_c(1:lastcp); NaN];
                    xp_c=[xp_c(1:lastcp); NaN];
                  else;
                    cp_c=[cp_c(1:lastcp); c(lasti:i-1); NaN];
                    xp_c=[xp_c(1:lastcp); x(lasti:i-1); NaN];
                  end;
                end;
             end;
             lastcp=size(xp_c,1);
             lasti=i;
             lastbreakt=i;
             vlinex_c{nbreaks_c}=[x(i-1) x(i-1)];
             vliney_c{nbreaks_c}=[c(i-1) c(i)];
             hlinex_c{nbreaks_c}=[x(i-1) x(i)];
             hliney_c{nbreaks_c}=[c(i) c(i)];
         end; 
      end;
   end;
   if (nbreaks_c);
     cp_c=[cp_c(1:lastcp); c(lasti:i)];
     xp_c=[xp_c(1:lastcp); x(lasti:i)];
   else;
     cp_c=c(i)*ones(T+1,1);
     xp_c=x;
   end;

   xp_c1=[];
   cp_c1=[];
   nbreaks_c1=0;
   lastbreakt=0;
   lastcp=0;
   lasti=0;
   vlinex_c1=cell(0);
   vliney_c1=cell(0);
   hlinex_c1=cell(0);
   hliney_c1=cell(0);
   for i=1:T+1;
      if (i > 1);
         if (c1(i-1)>c1(i));
              nbreaks_c1=nbreaks_c1+1;
              if (nbreaks_c1 == 1);
                cp_c1=[c1(1:i-1); NaN];
                xp_c1=[x(1:i-1); NaN];
              else;
                if (lastbreakt == i-1);
                   cp_c1=[cp_c1(1:lastcp); c1(lastbreakt); NaN];
                   xp_c1=[xp_c1(1:lastcp); x(lastbreakt); NaN];
                else;
                  if (lasti == i-1);
                    cp_c1=[cp_c1(1:lastcp); NaN];
                    xp_c1=[xp_c1(1:lastcp); NaN];
                  else;
                    cp_c1=[cp_c1(1:lastcp); c1(lasti:i-1); NaN];
                    xp_c1=[xp_c1(1:lastcp); x(lasti:i-1); NaN];
                  end;
                end;
             end;
             lastcp=size(xp_c1,1);
             lasti=i;
             lastbreakt=i;
             vlinex_c1{nbreaks_c1}=[x(i-1) x(i-1)];
             vliney_c1{nbreaks_c1}=[c1(i-1) c1(i)];
             hlinex_c1{nbreaks_c1}=[x(i-1) x(i)];
             hliney_c1{nbreaks_c1}=[c1(i) c1(i)];
         end; 
      end;
   end;
   if (nbreaks_c1);
     cp_c1=[cp_c1(1:lastcp); c1(lasti:i)];
     xp_c1=[xp_c1(1:lastcp); x(lasti:i)];
   else;
     cp_c1=c1(i)*ones(T+1,1);
     xp_c1=x;
   end;

   xp_c2=[];
   cp_c2=[];
   nbreaks_c2=0;
   lastcp=0;
   lastbreakt=0;
   lasti=0;
   vlinex_c2=cell(0);
   vliney_c2=cell(0);
   hlinex_c2=cell(0);
   hliney_c2=cell(0);
   for i=1:T+1;
      if (i > 1);
         if (c2(i-1)>c2(i));
              nbreaks_c2=nbreaks_c2+1;
              if (nbreaks_c2 == 1);
                cp_c2=[c2(1:i-1); NaN];
                xp_c2=[x(1:i-1); NaN];
              else;
                if (lastbreakt == i-1);
                  cp_c2=[cp_c2(1:lastcp); c2(lastbreakt); NaN];
                  xp_c2=[xp_c2(1:lastcp); x(lastbreakt); NaN];
                else;
                  if (lasti == i-1);
                    cp_c2=[cp_c2(1:lastcp); NaN];
                    xp_c2=[xp_c2(1:lastcp); NaN];
                  else;
                    cp_c2=[cp_c2(1:lastcp); c2(lasti:i-1); NaN];
                    xp_c2=[xp_c2(1:lastcp); x(lasti:i-1); NaN];
                  end;
                end;
             end;
             lastcp=size(xp_c2,1);
             lasti=i;
             lastbreakt=i;
             vlinex_c2{nbreaks_c2}=[x(i-1) x(i-1)];
             vliney_c2{nbreaks_c2}=[c2(i-1) c2(i)];
             hlinex_c2{nbreaks_c2}=[x(i-1) x(i)];
             hliney_c2{nbreaks_c2}=[c2(i) c2(i)];
         end; 
      end;
   end;
   if (nbreaks_c2);
     cp_c2=[cp_c2(1:lastcp); c2(lasti:i)];
     xp_c2=[xp_c2(1:lastcp); x(lasti:i)];
   else;
     cp_c2=c2(i)*ones(T+1,1);
     xp_c2=x;
   end;

   plot(ax,xp_c,cp_c,'k-','Linewidth',2);
   if (size(cp_c2,1) & size(cp_c1,1));
    plot(ax,xp_c1,cp_c1,'r-','Linewidth',2);
    if (size(cp_c2,1) == size(cp_c1,1) & sum(cp_c2(~isnan(cp_c2))-cp_c1(~isnan(cp_c1))) ~= 0);
       plot(ax,xp_c2,cp_c2,'b-','Linewidth',2);
       %legend('c','c1','c2','Location','Northeast');
    else;
%     if (size(cp_c2,1));
%       plot(xp_c2,cp_c2,'b-','Linewidth',2);
%       legend('c','c2','Location','Northeast');
%     end;
%     if (size(cp_c1,1));
%       plot(xp_c1,cp_c1,'r-','Linewidth',2);
%       legend('c','c1','Location','Northeast');
%     end;
    end;
   end;
   for i=1:nbreaks_c;
     line(ax,vlinex_c{i},vliney_c{i},'Color','k','Linewidth',2);
     line(ax,hlinex_c{i},hliney_c{i},'Color','k','Linewidth',2);
   end;
   for i=1:nbreaks_c1;
     line(ax,vlinex_c1{i},vliney_c1{i},'Color','r','Linewidth',2);
     line(ax,hlinex_c1{i},hliney_c1{i},'Color','r','Linewidth',2);
   end;
   for i=1:nbreaks_c2;
     line(ax,vlinex_c2{i},vliney_c2{i},'Color','b','Linewidth',2);
     line(ax,hlinex_c2{i},hliney_c2{i},'Color','b','Linewidth',2);
   end;
   xlabel(ax,'Period, t');
   ylabel(ax,'Price and marginal costs');

   % remove nans from the firm price trajectories to fill in regions where one firm's price
   % is lower than the other's and fill the regions with red and blue to indicate graphically
   % the profits earned.

   xp_c1=xp_c1(~isnan(xp_c1));
   cp_c1=cp_c1(~isnan(cp_c1));
   xp_c2=xp_c2(~isnan(xp_c2));
   cp_c2=cp_c2(~isnan(cp_c2));

   last_leapfrog_date=1;
   nleapfrogs=sum(cp_c1 ~= cp_c2);
   if (nleapfrogs);
      leapfrog_dates=[];
      last_leapfrog_date=1;
      while (last_leapfrog_date < T+1);
        last_leapfrog_date=min(find(cp_c1 ~= cp_c2 & (cp_c1 < cp_c1(last_leapfrog_date) | cp_c2 < cp_c2(last_leapfrog_date))));
        if (size(last_leapfrog_date,1));
        leapfrog_dates=[leapfrog_dates; last_leapfrog_date];
        end;
      end;
      nleapfrogs=size(leapfrog_dates,1);
   end;

   if (autoplay);
   title({sprintf('Simulated play, \\color{red} firm 1 profits %g \\color{blue} firm 2 profits %g',cumprof1,cumprof2),...
         sprintf('\\color{black} %i leapfrog investments (shaded areas are gross profits, ignoring investment costs)',nleapfrogs)});
   else;
   title({sprintf('Simulated play, \\color{red} firm 1 profits %g \\color{blue} your (firm 2) profits %g',cumprof1,cumprof2),...
         sprintf('\\color{black} %i leapfrog investments (shaded areas are gross profits, ignoring investment costs)',nleapfrogs)});
   end;

   for i=1:nleapfrogs;
       lfbd=leapfrog_dates(i);
       if (cp_c1(lfbd) < cp_c2(lfbd));
          lfed=lfbd+1;
          while (cp_c1(lfed) == cp_c1(lfbd) & cp_c2(lfed) == cp_c2(lfbd) & lfed <= T);
             lfed=lfed+1;
          end;
          if (shade_profits);
          h=fill([lfbd-.99 lfbd-.99 lfed-1 lfed-1],[cp_c1(lfbd) cp_c2(lfbd) cp_c2(lfbd) cp_c1(lfbd)],[1  0.5  .05],'facealpha',.25);
          uistack(h,'bottom'); %to prevent the fill to run over the lines
          end;
          fprintf('leapfrog %i c1=%g c2=%g at t=%i until t=%i\n',i,cp_c1(lfbd),cp_c2(lfbd),lfbd-1,lfed-1);
           
       else;
          lfed=lfbd+1;
          while (cp_c1(lfed) == cp_c1(lfbd) & cp_c2(lfed) == cp_c2(lfbd) & lfed <= T);
             lfed=lfed+1;
          end;
          if (shade_profits);
          h=fill([lfbd-.99 lfbd-.99 lfed-1 lfed-1],[cp_c2(lfbd) cp_c1(lfbd) cp_c1(lfbd) cp_c2(lfbd)],[0.5 0.5 1],'facealpha',.25);
          uistack(h,'bottom'); %to prevent the fill to run over the lines
          end;
          fprintf('leapfrog %i c1=%g c2=%g at t=%i until t=%i\n',i,cp_c1(lfbd),cp_c2(lfbd),lfbd-1,lfed-1);
       end;
   end;

        
   % axis('tight');

%   if (~autoplay);   
%     probdiff=gamehist(:,4)-gamehist(:,12);
%     figure(2);
%     bar(probdiff);
%     title('Difference between your (firm 2) and equilibrium investment probabilities for firm 2');
%     ylabel('Equilibrium investment probability less your investment probability')
%     xlabel('Game period');
%   end;
   


