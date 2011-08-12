      PROGRAM FOO
      INTEGER I, J
      INTEGER A(10, 10)
      DO J = 1, 10
         DO I = 1, 10
            A(I,J) = I+J*10
         ENDDO
      ENDDO
      END
