C     Scalar replacement in the presence of conditional control flow
C
C     Steve Carr and Ken Kennedy
C
C     Software - Practive and Experience, Vol. 24, No. 1, pp. 51-77, 1994
C
C     Fourth example, page 10: PIPS profitability analysis not
C     convincing although the result is kind of OK. The exportation
C     statement could be avoided by using the last assignment

      subroutine scalarization33(a, b, c, d, e, f, n)
      real a(n), b(n), c(n), d(n), e(n), f(n)

      do i = 1,n
         if(a(i).gt.0.) then
            b(i) = c(i) + d(i)
         else
            f(i) = c(i) + d(i)
         endif
         c(i) = e(i) + b(i)
      enddo

      print *, c, f

      end
