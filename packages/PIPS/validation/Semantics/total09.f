      program total09

C     Check that executed loops are proprely handled: a condition
C     on N should be found. How about the second access? It's OK

      real a(10)

      do i = 1, 10
         a(n) = 0.
      enddo

      a(i-1) = 1.

      end
