      subroutine trusted_ref05(a, n, m)

C     Check that initial condition are propagated upwards at loops

      real a(n)

      do i = 1, m
         a(i) = 0.
      enddo

      end
