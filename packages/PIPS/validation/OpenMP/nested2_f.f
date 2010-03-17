      PROGRAM NESTED

      INTEGER K,I,J,SIZE
      PARAMETER (SIZE=100)
      REAL ARRAY(SIZE,SIZE)
      REAL ARRAY2(SIZE,SIZE,SIZE)

      DO I = 1, SIZE
         DO J = 1, SIZE
            DO K = 1, SIZE
               ARRAY(K,I) = I + K +J
            ENDDO
         ENDDO
      ENDDO

      DO I = 1, SIZE
         DO J = 1, SIZE
            DO K = 1, SIZE
               ARRAY2(K,J,I) = I + K +J
            ENDDO
         ENDDO
      ENDDO

      PRINT *, ARRAY
      PRINT *, ARRAY2
      END

