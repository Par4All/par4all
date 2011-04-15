C     Scalar replacement in the presence of conditional control flow
C
C     Steve Carr and Ken Kennedy
C
C     Software - Practive and Experience, Vol. 24, No. 1, pp. 51-77, 1994
C
C     Second example, which PIPS decides to be unprofitable

      subroutine scalarization31(a, b, c, d, e, m, n)
      real a(n), b(n), c(n), d(n), e(n), m(n)

      do i = 1,n
         if(m(i).lt.0.) a(i) = b(i) + c(i)
         d(i) = a(i) + e(i)
      enddo

      do i = 1,n
         if(m(i).lt.0.) a(i) = b(i) + c(i)
         d(i) = a(i) + e(i)
      enddo

      print *, d

      do i = 1,n
         if(m(i).lt.0.) a(i) = b(i) + c(i)
         d(i) = a(i) + e(i)
         d(i) = d(i) + a(i) * e(i)
      enddo

      end

