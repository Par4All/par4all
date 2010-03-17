      program doubleif

c     test of entry and exit node extraction by Ronan's restructurer

      j = 2

      if (j .eq. 3) then
         i = 5
      else
         i = 4
      endif
      
      if (i .eq. 4) then
         j = 6
      else
         j = 7
      endif

      print *, j

      end
