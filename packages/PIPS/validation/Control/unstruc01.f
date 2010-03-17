      program unstruc01

C     Check bug in controlizer when controlize_test() calls itself
C     recursively

      if (nass.eq.157) then
         nq=4
      elseif (nass.eq.60) then
         nq=1
      else
         return
      endif

      end
