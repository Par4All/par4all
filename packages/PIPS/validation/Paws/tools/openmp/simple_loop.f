      SUBROUTINE A1(A, N, B)

      INTEGER I, N
      REAL B(N), A(N)

      DO i=2,N
         B(I) = (A(I) + A(I - 1)) / 2.0
      ENDDO
 
      END
