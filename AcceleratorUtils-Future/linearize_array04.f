      PROGRAM FOO
      INTEGER I, J
      INTEGER A(6:15, 6:15)
      DO J = 6 ,15
         DO I = 6 ,15
            A(I,J) = I + J * 10
         enddo
      enddo
      END
