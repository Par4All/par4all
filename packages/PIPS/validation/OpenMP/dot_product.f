      PROGRAM DOT_PRODUCT

      INTEGER N, CHUNKSIZE, CHUNK, I
      PARAMETER (N=100)
      PARAMETER (CHUNKSIZE=10)
      REAL A(N), B(N), RESULT

!     Some initializations
      DO I = 1, N
         A(I) = I * 1.0
         B(I) = I * 2.0
      ENDDO
      RESULT= 0.0
      CHUNK = CHUNKSIZE

      DO I = 1, N
         RESULT = RESULT + (A(I) * B(I))
      ENDDO

      PRINT *, 'Final Result= ', RESULT
      END
