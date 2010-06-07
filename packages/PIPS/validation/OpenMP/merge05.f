      PROGRAM MERGE05
      INTEGER K,I,J,size
      parameter (size = 100)
      REAL array (size,size,size)

      DO I = 1, SIZE
         DO J = 1, SIZE
            DO K = 1, SIZE
               ARRAY(K,I,J) = I + K +J
            ENDDO
         ENDDO
      ENDDO

      print *, array

      END

