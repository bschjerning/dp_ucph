% h2(c1,c2,c): function to provide expected value to firm 2 in state (c1,c2,c)
%              John Rust, Georgetown University, January 2017

   function [h2v]=h2(c1,c2,stage);

   global vmat;

   h1v=0;

   % start at stage 1 and work up

   for cs=1:stage-1;

      if (size(vmat,1) > 0);
         
         i=find(vmat(:,1)==c2 & vmat(:,2) == c1 & vmat(:,3)==cs);

         if (size(i,1) > 0);

            v1=vmat(i,4);
            v2=vmat(i,5);

         else;

            [v1 v2 p1 p2]=stagegame_eq(c1,c2,cs);

         end;

      else;

        [v1 v2 p1 p2]=stagegame_eq(c1,c2,cs);

      end;

      h1v=h1v+v1*stp(cs,stage);

   end;

