C     Scalar replacement in the presence of conditional control flow
C
C     Steve Carr and Ken Kennedy
C
C     Software - Practive and Experience, Vol. 24, No. 1, pp. 51-77, 1994
C
C     Third example, page 6: the profitability analysis by PIPS is
C     simplistic because the control is not taken into account; also
C     references a(i) and a(i-1) prevent scalarization, which does not
C     matter much here

      subroutine scalarization32(a, b, c, d, e, n)
      real a(n), b(n), c(n), d(n), e(n)

      do i = 1,n
         if(b(i).lt.0.) then
            c(i) = a(i) + d(i)
         else
            c(i) = a(i-1) + d(i)
         endif
         e(i) = c(i) + a(i)
      enddo

      print *, e

      end

