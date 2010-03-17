      subroutine trusted_ref04(a, n, m)

C     Check that initial condition are propagated upwards at tests

      real a(n)

      if(x.gt.0) then
         do i = 2, m
            if(x.gt.eps) then
               a(i) = 0.
            else
               a(i-1) = 1.
            endif
         enddo
         print *, i
      endif

      print *, m

      end
