      program aa07

C     The vector expression is not prettyprinted because PIPS does not manage to
C     parallelize such a loop. Strict monotonicity of power over positive number
C     is not taken into account.

      real x(10), y(10), z(10), u(10,10)

      do i = 1, 3
         x(i**2+1) = 0.
      enddo

      end

