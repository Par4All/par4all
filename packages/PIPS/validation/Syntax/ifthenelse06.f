      program ifthenelse06

C     Labelled else which leads the SUN f77 compiler to raise a user warning
C     And initially to a user error for PIPS

      read *,i

      if(i.eq.0) then
         print *, i
         go to 1
      elseif(i.eq.1) then
         print *, i+1
 1    else
         print *,i-1
      endif

      end
