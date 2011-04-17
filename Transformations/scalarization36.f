C     Scalar replacement in the presence of conditional control flow
C
C     Steve Carr and Ken Kennedy
C
C     Software - Practive and Experience, Vol. 24, No. 1, pp. 51-77, 1994
C
C     Sixth example, page 24: Livermore Loop 6 (or 11) cannot be handled
C     by PIPS because we do not exploit dependence arcs and because
C     regions fuse here two independent references

      subroutine scalarization36(b, w, n)
      real b(n,n), w(n)

      do i = 2, n
         do k = 1, i-1
            w(i) = w(i)+b(i,k)*w(i-k)
         enddo
      enddo

      print *, w

      end

