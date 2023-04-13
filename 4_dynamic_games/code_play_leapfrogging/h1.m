% h1(c1,c2,c): function to provide expected value to firm 1 in state (c1,c2,c)
%              John Rust, Georgetown University, January 2017

   function [h1v]=h1(c1,c2,stage);

   global vmat;

   h1v=0;

   if (c1 >= c2);
      return;
   end;

   % start at stage 1 and work up

   for cs=1:stage-1;

      if (size(vmat,1) > 0);
         
         i=find(vmat(:,1)==c1 & vmat(:,2) == c2 & vmat(:,3)==cs);

         if (size(i,1) > 0);

            v1=vmat(i,4);
            v2=vmat(i,5);

         else;

            [p1 p2 v1 v2]=stagegame_eq(c1,c2,cs);

         end;

      else;

        [p1 p2 v1 v2]=stagegame_eq(c1,c2,cs);

      end;

      h1v=h1v+v1*stp(cs,stage);

   end;

