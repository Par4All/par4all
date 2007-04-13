      program ifthenelse01

      read *,i

      if(i.eq.0) then
         print *, i
 1    elseif(i.eq.1) then
         print *, i+1
      else
         print *,i-1
      endif

      end
