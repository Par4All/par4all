      program unstruc05

C     Check special case of unique successor

      if (nass.eq.157) then
c        true branch
         go to 100
      else
c        false branch
         go to 100
      endif

 100  print *, nass

      end
