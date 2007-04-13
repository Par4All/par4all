      program nullincr2

c     This is Fortran compatible. But a user error should be raised.

      real t(10)

      k = 0
      do i = 1, 10, k
         t(i) = 0.
      enddo

      end
