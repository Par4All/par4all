      PROGRAM NESTED_REDUCTION
      INTEGER K,I,SIZE
      PARAMETER (SIZE=100)
      REAL SUM

      SUM = 0

      DO I = 1, SIZE
         DO K = 1, SIZE
            SUM = SUM + I + K
         ENDDO
      ENDDO

      PRINT *, SUM
      END

