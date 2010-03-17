      program order
c     to check the ordering used to  print preconditions and transformers

      if(n.gt.1) then
         i = 2
         j = 3
      endif

      if(n.le.1) then
         j = 3
         i = 2
      endif

      i = 2
      j = 3
      j = 3
      i = 2

      x = 1.
      y = 2.

      k = j + m - i

      if( k - j .gt. i -2 ) then
         m = -5
         print *, m
      endif

      print *, i, j, k, m

      end

      subroutine toto

      k = j + m - i

      if( k - j .gt. i -2 ) then
         m = 5
      endif

      end
