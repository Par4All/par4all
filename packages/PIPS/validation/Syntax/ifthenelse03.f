      program ifthenelse03

C     Bug: do not core dump for user errors such as a double ELSE

      read *,i

      if(i.eq.0) then
         print *, i
      else
         print *, i+1
      else
         print *,i-1
      endif

      end
