      program dead4

C     Check that negation are properly interpreted

      i = 1
      j = 1
      if (i.eq.j) then
         k = 2
      else
         k = 4
      endif

      print *,i,j,k
      end
