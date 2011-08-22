C     Scalar replacement in the presence of conditional control flow
C
C     Steve Carr and Ken Kennedy
C
C     Software - Practive and Experience, Vol. 24, No. 1, pp. 51-77, 1994
C
C     Fifth example, page 18: b(k) is not scalarized by PIPS... because
C     we do not try to scalarize from outside a loop. We would also miss
C     opportunities within a basic block not included in a loop. Or
C     because we do not check that the reference is loop invariant and
C     that the profitability analysis is different, as well as the
C     importation and exportation of the initial and final values, if
C     any
C
C     b(k) is scalarized by the second scalarization algorithm
C     implemented in PIPS
C
C     Also, we do not try to carry a value from one iteration to the next

      subroutine scalarization34(a, b, c, d, e, m, n)
      real a(n), b(n), c(n), d(n), e(n), m(n)

      do i = 1, 100
         if(m(i).lt.0.) e(i) = c(i)
         a(i) = c(i) + d(i)
         b(k) = b(k)+a(i-1)
      enddo

      print *, b, e

      end

