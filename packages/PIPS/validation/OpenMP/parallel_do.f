      PROGRAM PARALLEL_DO

      INTEGER N, I
      PARAMETER (N=100)
      REAL A(N)

      DO I = 1, N
         A(I) = I * 1.0
      ENDDO
      END
