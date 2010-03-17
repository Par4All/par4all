!     Explicit boolean in Fortran test condition

      integer function if05()

      integer i, n

      i = 0

      if(n.gt.0) then
         i = i + 1
         n = n - 1
      else
         i = i - 1
         n = n + 1
      endif

      if04 = i

      end
