      PROGRAM NESTED

      INTEGER K,I,SIZE
      PARAMETER (SIZE=100)
      REAL ARRAY(SIZE,SIZE)

      DO I = 1, SIZE
         DO K = 1, SIZE
            ARRAY(K,I) = I + K
         ENDDO
      ENDDO

      PRINT *, ARRAY
      END

