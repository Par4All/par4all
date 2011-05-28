C     Scalar replacement in the presence of conditional control flow
C
C     Steve Carr and Ken Kennedy
C
C     Software - Practive and Experience, Vol. 24, No. 1, pp. 51-77, 1994
C
C     Sixth example, page 24: Livermore Loop 5 cannot be handled by PIPS
C     because we do not try to carry a value from one iteration to the next

      subroutine scalarization35(x, y, z, n)
      real x(n), y(n), z(n)

      do i = 2, n
         x(i) = z(i)*(y(i)-x(i-1))
      enddo

      print *, x

      end

