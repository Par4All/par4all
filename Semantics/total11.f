      subroutine total11(a, n)

C     Check that executed loops are proprely handled

      real a(n)

      read *, m

      if(m.lt.1) stop

      do i = 1, m
         a(i) = 0.
      enddo

      end
