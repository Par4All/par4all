      program modincr02

c     This is Fortran compatible. But the loop range cannot be
c     used in the precondition

      real t(10)

      do i = 1, n, k
         t(i) = 0.
         n = n + 1
      enddo

      do i = 1, n, 1
         t(i) = 0.
         n = n + 1
      enddo

      do i = 1, n, -1
         t(i) = 0.
         n = n + 1
      enddo

      end
