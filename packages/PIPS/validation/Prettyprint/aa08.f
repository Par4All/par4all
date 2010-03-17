      program aa08

C     The vector expression is not prettyprinted because PIPS does not manage to
C     parallelize such a loop. Strict monotonicity of power over positive numbers
C     is not taken into account.

      real u(10,10)

      do i = 1, 10
         do j = 1, 10
            u(i,j) = 0.
         enddo
      enddo

      end

