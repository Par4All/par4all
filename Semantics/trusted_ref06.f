      subroutine trusted_ref06(a, n, m)

C     Check that necessary initial conditions are propagated upwards at
C     loops

      real a(n)

      do i = 1, 10
         a(i) = 0.
      enddo

      end
