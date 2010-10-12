C     Example used by Khadija Immadoueddine, p. 7 in her report
C
C     It seems that she has unrolled the j loop by hand and the i loop
C     with PIPS. In fact, she performed an unroll-and-jam
C
      subroutine unroll13(n1, n2, a, b, c)
C
      real a(0:n1+1,0:n2+1)
      real b(0:n1+1,0:n2+1)
      real c(0:n1+1,0:n2+1)

      do 200 j = 1, n2
         do 300 i = 1, n1
            a(i,j)=a(i+1,j)*b(i,j)+a(i,j+1)*c(i,j)
 300     continue
 200  continue

      END
