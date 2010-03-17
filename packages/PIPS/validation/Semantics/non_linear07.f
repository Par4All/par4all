      subroutine non_linear07(t, m)

C     Check rules for multiplication and loop bounds

      real t(m)

      read *, n
      do i = n*n+2, m
         t(i) = 0.
      enddo

      print *, i

      END
