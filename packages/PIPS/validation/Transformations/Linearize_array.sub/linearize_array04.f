      PROGRAM FOO
      INTEGER I, J
      INTEGER A(5:10, 5:10)
      DO J = 5 ,10
         DO I = 5 ,10
            A(I,J) = I * J
         enddo
      enddo
      PRINT *,A
      END
