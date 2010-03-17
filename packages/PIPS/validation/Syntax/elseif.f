C Une suite de if ... else if ... else if .... endif ne parse pas
C
      subroutine elseif
      if (i.le.0) then
         j = 11
         if( i.lt.2) then
            j = 12
         else 
            if (i.lt.3) then
               j = 13
            endif
         endif
         do 10 i = 1,10
            if (x.ge.tprime) then
               j = 5
               if (kmodele.eq.1) then
                  j = 6
               else
                  if (kmodele.eq.0) then
                     j = 7
                  else
                     stop 
                  endif
               endif
            else
               j = 9
            endif
 10      continue
      else
         j = 10
      endif
      end
