      program ifthenelse05

      read *,i

      if(i.eq.0) then
         print *, i
         go to 1

 1    elseif(i.eq.1) then
         print *, i+1
      else
         print *,i-1
      endif

      end
