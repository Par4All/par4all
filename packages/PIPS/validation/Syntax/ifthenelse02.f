      program ifthenelse02

C     Bug EDF: comment before a labelled elseif

      read *,i

      if(i.eq.0) then
         print *, i

 1    elseif(i.eq.1) then
         print *, i+1
      else
         print *,i-1
      endif

      end
