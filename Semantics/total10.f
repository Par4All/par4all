      subroutine total10(a, n)

C     Check that may be executed loops are proprely handled and that
C     total postcondition is generated for subroutines in the
C     intraprocedural case

      real a(n)

      read *, m

      do i = 1, m
         a(i) = 0.
      enddo

      end
